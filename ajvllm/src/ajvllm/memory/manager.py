"""Request block tables, full-prefix reuse, copy-on-write and transactional reservations."""

import hashlib
import json
from dataclasses import dataclass, field

import torch

from ajvllm.config import MemoryConfig
from ajvllm.memory.blocks import BlockManager
from ajvllm.memory.storage import PagedBatch, PagedKVStorage


@dataclass
class SequenceState:
    blocks: list[int] = field(default_factory=list)  # list of block indices allocated to this sequence
    tokens: tuple[int, ...] = ()  # tokens already computed for this sequence
    hashes: list[bytes] = field(default_factory=list)
    salt: str = ""
    pending_hit_tokens: int = 0


class KVCacheManager:
    def __init__(self, config: MemoryConfig, *, layers, kv_heads, head_dim, device, dtype, max_model_len, max_num_seqs):
        blocks_per_request = (max_model_len + config.block_size - 1) // config.block_size
        num_blocks = config.num_blocks or blocks_per_request * max_num_seqs
        if num_blocks < blocks_per_request:
            raise ValueError("KV pool must fit at least one maximum-context request")
        self.storage = storage = PagedKVStorage(
            layers, num_blocks, config.block_size, kv_heads, head_dim, device=device, dtype=dtype
        )
        self.block_size = storage.block_size
        self.blocks = BlockManager(storage.tensor.shape[2])
        # all requests/sequences kv state
        self.states: dict[str, SequenceState] = {}
        self.enable_prefix_cache = config.enable_prefix_cache
        # Managers are private to an immutable model instance/dtype; pages never cross models.
        self.namespace = config.cache_namespace
        self.hits = 0
        self.hit_tokens = 0
        self.cow_copies = 0
        self.preemptions = 0
        self.written_tokens = 0
        self.peak_used_blocks = 0

    @property
    def capacity_tokens(self):
        return len(self.blocks.ref_counts) * self.block_size

    def _root(self, salt):
        return hashlib.sha256(json.dumps((self.namespace, salt), ensure_ascii=True).encode()).digest()

    def _hash(self, parent, tokens):
        return hashlib.sha256(parent + json.dumps(tokens, separators=(",", ":")).encode()).digest()

    def attach(self, request_id, tokens, salt="") -> int:
        """make a sequenceState for this request, and return the number of tokens already computed for this request"""
        if request_id in self.states:
            return len(self.states[request_id].tokens)
        state = SequenceState(salt=salt)
        parent = self._root(salt)
        if self.enable_prefix_cache:
            # KV does not contain final logits: always recompute at least one token.
            for start in range(0, (len(tokens) - 1) // self.block_size * self.block_size, self.block_size):
                block_tokens = tokens[start : start + self.block_size]
                key = self._hash(parent, block_tokens)
                block = self.blocks.prefixes.get(key)
                if block is None:
                    break
                self.blocks.retain(block)
                state.blocks.append(block)
                state.hashes.append(key)
                parent = key
            count = len(state.blocks) * self.block_size
            state.tokens = tuple(tokens[:count])
            state.pending_hit_tokens = count
        self.states[request_id] = state
        return len(state.tokens)

    def reserve(self, request_id, end_pos) -> bool:
        state = self.states[request_id]
        # needed new blocks num
        needed = (end_pos + self.block_size - 1) // self.block_size - len(state.blocks)
        copy_tail = bool(
            state.tokens
            and len(state.tokens) % self.block_size
            and self.blocks.ref_counts[state.blocks[len(state.tokens) // self.block_size]] > 1
            and end_pos > len(state.tokens)
        )
        # blockManager allocates new blocks
        allocated = self.blocks.allocate(max(0, needed) + copy_tail)
        if allocated is None:
            return False
        # COW: copy the last block if it is shared and we are extending the sequence
        if copy_tail:
            tail = len(state.tokens) // self.block_size
            old = state.blocks[tail]
            new = allocated[0]
            try:
                self.storage.copy_block(old, new)
            except Exception:
                for block in allocated:
                    self.blocks.release(block)
                raise
            allocated.pop(0)
            state.blocks[tail] = new
            self.blocks.release(old)
            self.cow_copies += 1
        state.blocks.extend(allocated)
        self.peak_used_blocks = max(self.peak_used_blocks, sum(ref > 0 for ref in self.blocks.ref_counts))
        return True

    def fork(self, source, target):
        if target in self.states:
            raise ValueError("fork target already exists")
        source = self.states[source]
        blocks = source.blocks[: (len(source.tokens) + self.block_size - 1) // self.block_size]
        for block in blocks:
            self.blocks.retain(block)
        self.states[target] = SequenceState(blocks, source.tokens, list(source.hashes), source.salt)

    def commit(self, request_id, tokens):
        state = self.states[request_id]
        self.hit_tokens += state.pending_hit_tokens
        self.hits += bool(state.pending_hit_tokens)
        state.pending_hit_tokens = 0
        self.written_tokens += len(tokens)
        state.tokens += tuple(tokens)
        if self.enable_prefix_cache:
            parent = state.hashes[-1] if state.hashes else self._root(state.salt)
            for index in range(len(state.hashes), len(state.tokens) // self.block_size):
                key = self._hash(parent, state.tokens[index * self.block_size : (index + 1) * self.block_size])
                self.blocks.publish(state.blocks[index], key)
                state.hashes.append(key)
                parent = key

    def release(self, request_id):
        state = self.states.pop(request_id, None)
        if state is not None:
            for block in reversed(state.blocks):
                self.blocks.release(block)

    def clear_prefix_cache(self):
        self.blocks.clear_prefixes()

    def batch(self, request_ids, positions, sequence_ids, contexts, *, gather=True) -> PagedBatch:
        """
        suppose block_size=4;
        suppose 3 requests: A prefill [A0 A1 A2 A3 A4 A5], 
                            B prefill [B0 B1 B2],
                            C decode [C5] with cached [C0 C1 C2 C3 C4];
        suppose block_A = [2,5], block_B = [7], block_C = [1,6]

        input: request_ids=[A0 A1 A2 A3 A4 A5 | B0 B1 B2 | C5], 
                positions=[0 1 2 3 4 5 | 0 1 2 | 5], 
                sequence_ids=[0 0 0 0 0 0 | 1 1 1 | 2], 
                contexts=[6,3,6]

        variable:
            width = 2,
            tables = [[2,5], [7,0], [1,6]],
            slots = [8 9 10 11 20 21 | 28 29 30 | 25],
            read_slots = [[8 9 10 11 20 21], [28 29 30 31 * *], [4, 5, 6, 7, 24, 25]],
            valid = [[1 1 1 1 1 1], [1 1 1 0 0 0], [1 1 1 1 1 1]]
        """
        device = self.storage.tensor.device
        # max number of blocks needed for any request
        width = (max(contexts) + self.block_size - 1) // self.block_size
        # physical block indices for each request, padded with 0s to the max width
        tables = [
            self.states[rid].blocks[:width] + [0] * max(0, width - len(self.states[rid].blocks)) for rid in request_ids
        ]
        tables = torch.tensor(tables, device=device, dtype=torch.long)
        # each token's physical slot in the storage tensor: block_index * block_size + position_in_block
        slots = tables[sequence_ids, positions // self.block_size] * self.block_size + positions % self.block_size
        if not gather:  # triton
            return PagedBatch(self.storage, tables, slots, None, None)
        offsets = torch.arange(max(contexts), device=device)
        # each request's all physical slots for reading, padded with 0s to the max width
        read_slots = tables[:, offsets // self.block_size] * self.block_size + offsets % self.block_size
        # each request's valid slots for reading
        valid = offsets[None, :] < torch.tensor(contexts, device=device)[:, None]
        return PagedBatch(self.storage, tables, slots, read_slots, valid)

    def snapshot(self):
        used = sum(ref > 0 for ref in self.blocks.ref_counts)
        block_bytes = self.storage.nbytes // len(self.blocks.ref_counts)
        return {
            "backend": "paged",
            "block_size": self.block_size,
            "num_blocks": len(self.blocks.ref_counts),
            "used_blocks": used,
            "free_blocks": len(self.blocks.free),
            "cached_blocks": len(self.blocks.prefixes),
            "shared_blocks": sum(ref > 1 for ref in self.blocks.ref_counts),
            "pool_bytes": self.storage.nbytes,
            "used_bytes": used * block_bytes,
            "peak_used_bytes": self.peak_used_blocks * block_bytes,
            "prefix_hits": self.hits,
            "prefix_hit_tokens": self.hit_tokens,
            "evictions": self.blocks.evictions,
            "cow_copies": self.cow_copies,
            "preemptions": self.preemptions,
            "written_tokens": self.written_tokens,
            "kv_write_bytes": self.written_tokens * block_bytes // self.block_size,
            "cow_copy_bytes": self.cow_copies * block_bytes,
        }
