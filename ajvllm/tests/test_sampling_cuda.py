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
