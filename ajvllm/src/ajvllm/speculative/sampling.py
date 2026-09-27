"""CUDA probability distributions and the standard speculative rejection rule."""

import torch


def uniforms(requests, count, device):
    draws = []
    for request in requests:
        if request.rng is None:
            request.rng = torch.Generator(device=device)
            if request.sampling_params.seed is None:
                request.rng.seed()
            else:
                request.rng.manual_seed(request.sampling_params.seed % (1 << 64))
        draws.append(torch.rand(count, device=device, generator=request.rng))
    return torch.stack(draws)


def categorical(probabilities, draws):
    cdf = probabilities.cumsum(-1)
    # Normalization roundoff must not select a zero-probability tail.
    mass = cdf[..., -1]
    target = torch.minimum(draws * mass, torch.nextafter(mass, torch.zeros_like(mass)))
    tokens = torch.searchsorted(cdf.contiguous(), target[..., None].contiguous(), right=True).squeeze(-1)
    return tokens.clamp_max(probabilities.shape[-1] - 1)


def verify(target, draft, proposals, draws):
    """Return GPU token/logprob rows and the accepted prefix length.

    target: (K+1,V), draft: (K,V), proposals: (K,), draws: K+1.
    All accepted => bonus from target[K]. First rejection => normalized (p-q)+.
    """
    single = target.ndim == 2
    if single:
        target, draft, proposals, draws = (x.unsqueeze(0) for x in (target, draft, proposals, draws))
    batch, k = proposals.shape
    rows = torch.arange(batch, device=target.device)
    p = target[:, :k].gather(-1, proposals[..., None]).squeeze(-1)
    q = draft.gather(-1, proposals[..., None]).squeeze(-1)
    accepted = (draws[:, :k] * q < p).to(torch.long).cumprod(-1).sum(-1)
    padded = torch.cat((draft, torch.zeros_like(draft[:, :1])), dim=1)
    residual = (target[rows, accepted] - padded[rows, accepted]).clamp_min(0)
    residual /= residual.sum(-1, keepdim=True)
    replacement = categorical(residual, draws[:, -1])
    tokens = torch.cat((proposals, replacement[:, None]), dim=1).clone()
    tokens[rows, accepted] = replacement
    logprobs = target.gather(-1, tokens[..., None]).squeeze(-1).log()
    result = torch.stack((tokens.double(), logprobs.double()), -1)
    return (result[0], accepted[0]) if single else (result, accepted)
