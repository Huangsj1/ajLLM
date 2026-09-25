"""General CUDA sampling semantics, transfer boundaries and independent RNG streams."""

import math
from unittest.mock import patch

import pytest
import torch

from ajvllm import SamplingParams
from ajvllm.requests import Request
from ajvllm.sampling.sampler import Sampler

pytestmark = pytest.mark.cuda


def request(rid="a", prompt=(0,), generated=(), **params):
    return Request(rid, prompt, SamplingParams(**params), 0, output_token_ids=list(generated))


@pytest.mark.parametrize("temperature", [0, 1.0, 0.5, 1e-100])
def test_temperature_and_selected_logprob(temperature):
    req = request(temperature=temperature, seed=10)
    logits = torch.tensor([0.0, 1.0, 2.0], device="cuda")
    with patch("torch.rand", return_value=torch.tensor([0.1], device="cuda")):
        sample = Sampler().sample({"a": logits}, [req])["a"]
    assert sample.token_id == 2
    expected = 0 if temperature < 1e-20 else -math.log(1 + math.exp(-1 / temperature) + math.exp(-2 / temperature))
    assert sample.logprob == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "top_k,top_p,allowed", [(0, 1, 4), (2, 1, 2), (8, 1, 4), (0, 0.5, 2), (3, 0.5, 2), (0, 1e-100, 1)]
)
def test_top_k_top_p_ties_and_boundary(top_k, top_p, allowed):
    req = request(temperature=1, top_k=top_k, top_p=top_p, seed=2)
    scores = torch.zeros(4, device="cuda")
    with patch("torch.rand", return_value=torch.tensor([0.99], device="cuda")):
        result = Sampler().sample({"a": scores}, [req])["a"]
    assert result.token_id == allowed - 1
    assert result.logprob == pytest.approx(-math.log(allowed), abs=1e-6)


def test_penalties_counts_stop_mask_and_mixed_policies():
    requests = [
        request(
            "penalty",
            prompt=(1,),
            generated=(2, 2),
            temperature=0,
            repetition_penalty=2,
            presence_penalty=1,
            frequency_penalty=0.5,
        ),
        request(
            "mask",
            prompt=(1,),
            generated=(2, 2),
            temperature=0,
            min_tokens=3,
            repetition_penalty=2,
            presence_penalty=1,
            frequency_penalty=0.5,
        ),
        request("ignore", temperature=1, min_tokens=1, ignore_eos=True, top_k=1),
        request("stop", temperature=1, min_tokens=1, ignore_eos=True, stop_token_ids=(2,), top_k=1),
    ]
    rows = [torch.tensor([0.0, 6.0, 8.0, 3.0], device="cuda") for _ in requests]
    rows[2] = rows[3] = torch.tensor([0.0, 1.0, 9.0, 2.0], device="cuda")
    original = rows[0].clone()
    result = Sampler().sample(dict(zip([r.request_id for r in requests], rows, strict=True)), requests, (1, 2))
    assert [result[r.request_id].token_id for r in requests] == [1, 3, 2, 3]
    torch.testing.assert_close(rows[0], original)


@pytest.mark.parametrize("values", [[0, float("nan"), 1], [0, float("inf"), 1], [-float("inf")] * 3])
def test_invalid_distribution_fails_entire_batch(values):
    requests = [request("valid", seed=0), request("invalid", seed=0)]
    with pytest.raises(ValueError, match="invalid or all-masked"):
        Sampler().sample(
            {"valid": torch.zeros(3, device="cuda"), "invalid": torch.tensor(values, device="cuda")}, requests
        )


def test_all_masked_and_negative_infinity():
    req = request(min_tokens=1, stop_token_ids=(0, 1, 2))
    with pytest.raises(ValueError, match="all-masked"):
        Sampler().sample({"a": torch.zeros(3, device="cuda")}, [req])
    req = request(temperature=0.7, seed=2)
    result = Sampler().sample({"a": torch.tensor([-torch.inf, 2.0, -torch.inf], device="cuda")}, [req])["a"]
    assert result.token_id == 1 and result.logprob == 0


def test_request_rng_independent_of_batch_order_and_membership():
    sampler = Sampler()
    scores = torch.zeros(11, device="cuda")

    def run(mixed):
        a = request("a", temperature=0.8, top_k=7, top_p=0.9, seed=123)
        b = request("b", temperature=0.8, seed=99)
        selected = []
        for step in range(12):
            ready = ([b, a] if step % 2 else [a, b]) if mixed else [a]
            results = sampler.sample({r.request_id: scores for r in ready}, ready)
            selected.append(results["a"].token_id)
        assert a.rng.device.type == "cuda"
        return selected

    assert run(False) == run(True)


def test_only_compact_samples_cross_to_cpu():
    sampler = Sampler()
    requests = [request("a", temperature=0.9, top_p=0.8), request("b", temperature=0)]
    rows = {r.request_id: torch.zeros(97, device="cuda") for r in requests}
    shapes = []
    original = torch.Tensor.cpu

    def transfer(tensor, *args, **kwargs):
        shapes.append(tuple(tensor.shape))
        return original(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, "cpu", transfer):
        result = sampler.sample(rows, requests)
    assert set(result) == {"a", "b"} and shapes == [(2, 3)]
    assert sampler.transfer_bytes == 48
    assert sampler.sample({}, []) == {}


def reference_distribution(logits, req, eos=()):
    """CUDA oracle matching stable top-k followed by inclusive nucleus filtering."""
    p = req.sampling_params
    scores = logits.float().clone()
    seen = torch.zeros_like(scores, dtype=torch.bool)
    seen[torch.tensor(req.token_ids, device="cuda", dtype=torch.long)] = True
    counts = torch.bincount(
        torch.tensor(req.output_token_ids, device="cuda", dtype=torch.long), minlength=scores.numel()
    )
    scores = torch.where(
        seen, torch.where(scores > 0, scores / p.repetition_penalty, scores * p.repetition_penalty), scores
    )
    scores = scores - p.presence_penalty * (counts > 0) - p.frequency_penalty * counts
    if len(req.output_token_ids) < p.min_tokens:
        for token in p.effective_stop_ids(eos):
            if token < scores.numel():
                scores[token] = -torch.inf
    values, ids = scores.sort(descending=True, stable=True)
    limit = 1 if p.temperature == 0 else p.top_k or scores.numel()
    temp = max(p.temperature if p.temperature else 1.0, torch.finfo(torch.float32).tiny)
    scaled = (values - values[0]) / temp
    scaled[limit:] = -torch.inf
    probs = scaled.softmax(0)
    previous = torch.cat((torch.zeros(1, device="cuda"), probs.cumsum(0)[:-1]))
    if p.top_p < 1:
        probs[(previous >= p.top_p) & (torch.arange(probs.numel(), device="cuda") > 0)] = 0
    probs /= probs.sum()
    return ids, probs


@pytest.mark.parametrize("vocab", [31, 1023, 1024, 1025, 2053, 151936])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tiled_sampling_against_cuda_oracle(vocab, dtype):
    torch.manual_seed(432)
    scores = torch.randn(4, vocab, device="cuda", dtype=dtype) * 3
    requests = [
        request("a", top_p=0.9, temperature=0.8, seed=1),
        request(
            "b",
            top_k=17,
            top_p=0.5,
            temperature=1.3,
            seed=2,
            prompt=(1, 1, 2),
            generated=(3, 3, 4),
            repetition_penalty=1.2,
            frequency_penalty=0.7,
        ),
        request(
            "c",
            top_p=1,
            temperature=0.4,
            seed=3,
            generated=(0, 0, 2),
            presence_penalty=-0.4,
            min_tokens=4,
            stop_token_ids=(1, 2),
            ignore_eos=True,
        ),
        request("d", top_k=vocab + 1, top_p=1e-100, temperature=0.9, seed=4),
    ]
    draws = [0.0, 0.11, 0.65, 0.99999]
    with patch("torch.rand", side_effect=[torch.tensor([u], device="cuda") for u in draws]):
        result = Sampler().sample(dict(zip([r.request_id for r in requests], scores, strict=True)), requests)
    for row, req, draw in zip(scores, requests, draws, strict=True):
        ids, probs = reference_distribution(row, req)
        cdf = probs.cumsum(0)
        index = torch.searchsorted(cdf / cdf[-1], torch.tensor(draw, device="cuda"), right=True)
        sample = result[req.request_id]
        assert sample.token_id == ids[index].item()
        assert sample.logprob == pytest.approx(probs[index].log().item(), abs=2e-5)


@pytest.mark.parametrize("vocab", [1024, 1025, 2053, 151936])
def test_nucleus_at_tile_boundaries(vocab):
    req = request(top_p=1024 / vocab, temperature=1, seed=1)
    with patch("torch.rand", return_value=torch.tensor([0.99999], device="cuda")):
        result = Sampler().sample({"a": torch.zeros(vocab, device="cuda")}, [req])["a"]
    # Compare the cutoff to high-precision arithmetic on the actual FP32 policy.
    # The old normalize-then-scan path can round an exact boundary differently.
    p = torch.tensor(req.sampling_params.top_p, device="cuda", dtype=torch.float32).double()
    retained = (p * vocab).ceil().clamp(1, vocab)
    draw = torch.tensor(0.99999, device="cuda", dtype=torch.float32).double()
    assert result.token_id == (draw * retained).floor().item()
    assert result.logprob == pytest.approx(-retained.log().item(), abs=2e-5)


def test_incremental_history_eviction_rebuild_and_release():
    sampler = Sampler(max_histories=1)
    a = request("a", prompt=(1, 1), generated=(2, 2), repetition_penalty=1.5, frequency_penalty=0.5, seed=0)
    b = request("b", prompt=(3,), generated=(4,), presence_penalty=0.5, seed=1)
    logits = torch.arange(19, device="cuda", dtype=torch.float32)
    for ready in ([a], [a], [b], [a, b], [a]):
        with patch("torch.rand", side_effect=[torch.tensor([0.25], device="cuda") for _ in ready]):
            result = sampler.sample({r.request_id: logits for r in ready}, ready)
        for r in ready:
            ids, probs = reference_distribution(logits, r)
            cdf = probs.cumsum(0)
            index = torch.searchsorted(cdf / cdf[-1], torch.tensor(0.25, device="cuda"), right=True)
            assert result[r.request_id].token_id == ids[index].item()
            r.output_token_ids.append(result[r.request_id].token_id)
        assert len(sampler.histories) <= 1
    sampler.release(a)
    assert not sampler.histories
    # Identity, rather than a reused public request ID, owns the history.
    replacement = request("a", prompt=(0,), repetition_penalty=2, temperature=0)
    sampler.sample({"a": logits}, [replacement])
    assert id(replacement) in sampler.histories and id(a) not in sampler.histories


def test_neutral_penalties_and_expired_stop_mask_do_not_retain_history():
    sampler = Sampler()
    req = request(min_tokens=1, stop_token_ids=(2,), seed=1)
    logits = torch.zeros(3, device="cuda")
    result = sampler.sample({"a": logits}, [req])["a"]
    assert len(sampler.histories) == 1 and result.token_id != 2
    req.output_token_ids.append(result.token_id)
    sampler.sample({"a": logits}, [req])
    assert not sampler.histories


@pytest.mark.parametrize("top_p", [0.9, 1.0])
def test_near_one_draw_never_selects_zero_mass_tail(top_p):
    torch.manual_seed(823)
    reqs = [request(str(i), top_k=1025, top_p=top_p, temperature=0.1) for i in range(4)]
    logits = torch.randn(4, 151936, device="cuda")
    logits[:, 2048:] = -torch.inf
    draw = torch.nextafter(torch.tensor([1.0], device="cuda"), torch.tensor([0.0], device="cuda"))
    with patch("torch.rand", return_value=draw):
        result = Sampler().sample(dict(zip([r.request_id for r in reqs], logits, strict=True)), reqs)
    for row, req in zip(logits, reqs, strict=True):
        ids, probs = reference_distribution(row, req)
        sample = result[req.request_id]
        assert sample.token_id in ids[probs > 0].tolist()
        assert math.isfinite(sample.logprob)


def test_history_cleanup_when_request_is_dropped():
    import gc

    sampler = Sampler()
    req = request(repetition_penalty=1.1)
    sampler.sample({"a": torch.zeros(7, device="cuda")}, [req])
    assert len(sampler.histories) == 1
    del req
    gc.collect()
    assert not sampler.histories
