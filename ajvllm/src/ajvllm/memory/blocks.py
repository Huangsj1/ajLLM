"""Physical block ownership and LRU reuse; no tensors or model dependencies."""

from collections import OrderedDict


class BlockManager:
    def __init__(self, num_blocks: int):
        # Reference counts indicate how many sequences use each physical block.
        self.ref_counts = [0] * num_blocks
        self.free = OrderedDict.fromkeys(range(num_blocks))  # Free block IDs, ordered by LRU.
        self.keys: list[bytes | None] = [None] * num_blocks  # Block ID -> prefix hash.
        self.prefixes: dict[bytes, int] = {}  # Prefix hash -> cached block ID.
        self.evictions = 0
        self.shared_blocks = 0

    def allocate(self, count: int) -> list[int] | None:
        # Preflight the entire reservation; capacity failure changes no ownership.
        if count > len(self.free):
            return None
        result = []
        for _ in range(count):
            # pop the first free block
            block, _ = self.free.popitem(last=False)
            key = self.keys[block]
            if key is not None:
                self.prefixes.pop(key, None)
                self.keys[block] = None
                self.evictions += 1
            self.ref_counts[block] = 1
            result.append(block)
        return result

    def retain(self, block: int):
        if self.ref_counts[block] == 0:
            del self.free[block]
        if self.ref_counts[block] == 1:
            self.shared_blocks += 1
        self.ref_counts[block] += 1

    def release(self, block: int):
        if self.ref_counts[block] <= 0:
            raise RuntimeError("block released without an owner")
        if self.ref_counts[block] == 2:
            self.shared_blocks -= 1
        self.ref_counts[block] -= 1
        if self.ref_counts[block] == 0:
            # put the block back to the free list
            self.free[block] = None
            # Uncached private tails are reused before cached prefix blocks.
            if self.keys[block] is None:
                # move the block to the front of the free list
                self.free.move_to_end(block, last=False)

    def publish(self, block: int, key: bytes):
        if key not in self.prefixes:
            self.keys[block] = key
            self.prefixes[key] = block

    def clear_prefixes(self):
        self.prefixes.clear()
        self.keys[:] = [None] * len(self.keys)
