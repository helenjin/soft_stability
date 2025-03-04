import torch
import math


def sample_alpha_pertbs(
    alpha: torch.LongTensor,
    radius: int,
    num_samples: int
):
    """
    Sample uniformly from:
        Delta_r = {alpha' : alpha' >= alpha, |alpha' - alpha| <= r}

    Args:
        alpha (torch.LongTensor): The 0/1-valued tensor of shape (n,).
        radius (int): The radius within which we sample.
        num_samples (int): The number of perturbed samples to generate.

    Returns:
        torch.LongTensor: Sampled perturbations of shape (num_samples, n).
    """
    original_shape = alpha.shape
    alpha = alpha.view(-1)

    samples = alpha.view(1, -1).repeat(num_samples, 1)

    # Find indices where alpha is 0 (these can potentially be flipped to 1)
    zero_indices = torch.nonzero(alpha == 0, as_tuple=False).squeeze()
    num_zeros = zero_indices.numel()
    if radius > num_zeros:
        raise ValueError(f"Radius {radius} > num zeros {num_zeros}")

    # Compute log-binomial coefficients in log-space, because we have massive blow-up otherwise.
    log_flip_probs = torch.tensor(
        [
            math.lgamma(num_zeros + 1) - math.lgamma(i + 1) - math.lgamma(num_zeros - i + 1)
            for i in range(radius + 1)
        ],
        dtype = torch.float32,
        device = alpha.device
    )

    # Convert log-probs to sampling-friendly format via Gumbel-max trick
    gumbel_noise = -torch.log(-torch.log(torch.rand(num_samples, radius + 1, device=alpha.device)))
    log_probs_with_noise = log_flip_probs.view(1, -1) + gumbel_noise
    num_flips = torch.argmax(log_probs_with_noise, dim=-1)

    # Select random indices to flip in each sample
    for i, flips in enumerate(num_flips):
        flip_inds = torch.randperm(num_zeros)[:flips]
        samples[i, zero_indices[flip_inds]] = 1 # Flip selected indices from 0 to 1

    return samples.view(num_samples, *original_shape).long()


@torch.no_grad()
def soft_stability_rate(
    f,
    x: torch.FloatTensor,
    alpha: torch.LongTensor,
    radius: int,
    epsilon: float = 0.1,
    delta: float = 0.1,
    batch_size: int = 16,
    return_all: bool = False,
):
    """
    Measure the soft stability rate for a classifier of form y = f(x, alpha), where:

        soft_stability_rate = Pr_{alpha' ~ Delta_r} [f(x, alpha') == f(x, alpha)]

    Args:
        f: Any function (ideally nn.Module) that takes as input x, alpha.
        x: The input to f of some shape. NOT batched.
        alpha: The 0/1 attribution of some shape. NOT batched.
        radius: The radius to which we give the guarantee.
        epsilon: The error tolerance.
        delta: The admissible failure probability
        batch_size: The batch size in case we run out-of-memory.
        return_all: Return all the intermediate information in a dictionary.

    Returns:
        soft_stability_rate: A value between 0 and 1.
    """
    C, H, W = x.shape
    y = f(x.view(1,C,H,W), alpha.unsqueeze(0)) # Reference prediction

    N = int(math.log(2/delta) / (2 * (epsilon**2))) + 1
    all_y_pertbs = []

    for alpha_pertbs in torch.split(sample_alpha_pertbs(alpha, radius, N), batch_size):
        repeat_pattern = [1] * (1 + x.ndim)
        repeat_pattern[0] = alpha_pertbs.size(0)
        y_pertbs = f(x.unsqueeze(0).repeat(*repeat_pattern), alpha_pertbs)
        all_y_pertbs.append(y_pertbs)

    all_y_pertbs = torch.cat(all_y_pertbs, dim=0)
    soft_stab_rate = (y.argmax(dim=-1) == all_y_pertbs.argmax(dim=-1)).float().mean()

    if return_all:
        return {
            "soft_stability_rate": soft_stab_rate,
            "y": y[0],
            "y_pertbs": all_y_pertbs,
        }

    else:
        return soft_stab_rate

    
    
def soft_stability_rate_text(
    f,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    alpha: torch.LongTensor,
    radius: int,
    epsilon: float = 0.1,
    delta: float = 0.1,
    batch_size: int = 16,
    return_all: bool = False,
):
    """
    Measure the soft stability rate for a classifier of form y = f(x, alpha), where:

        soft_stability_rate = Pr_{alpha' ~ Delta_r} [f(x, alpha') == f(x, alpha)]
    and x = input_ids, attention_mask.

    Args:
        f: Any function (ideally nn.Module) that takes as input x, alpha.
        input_ids: Shape (L,), this is NOT batched.
        attention_mask: Shape (L,), also NOT batched.
        alpha: The 0/1 attribution of some shape. NOT batched.
        radius: The radius to which we give the guarantee.
        epsilon: The error tolerance.
        delta: The admissible failure probability
        batch_size: The batch size in case we run out-of-memory.
        return_all: Return all the intermediate information in a dictionary.

    Returns:
        soft_stability_rate_text: A value between 0 and 1.
    """
    input_ids = input_ids.view(1,-1)
    attention_mask = attention_mask.view(1,-1)
    alpha = alpha.view(-1)

    # Reference prediction
    y = f(input_ids=input_ids, attention_mask=attention_mask, alpha=alpha.view(1,-1))

    N = int(math.log(2/delta) / (2 * (epsilon**2))) + 1
    all_y_pertbs = []

    for alpha_pertbs in torch.split(sample_alpha_pertbs(alpha, radius, N), batch_size):
        # input_ids and attention_mask can share the same repeat pattern
        repeat_pattern = [alpha_pertbs.size(0), 1]
        y_pertbs = f(
            input_ids = input_ids.view(1,-1).repeat(*repeat_pattern),
            attention_mask = attention_mask.view(1,-1).repeat(*repeat_pattern),
            alpha = alpha_pertbs
        )
        all_y_pertbs.append(y_pertbs)

    all_y_pertbs = torch.cat(all_y_pertbs, dim=0)
    soft_stab_rate = (y.argmax(dim=-1) == all_y_pertbs.argmax(dim=-1)).float().mean()

    if return_all:
        return {
            "soft_stability_rate": soft_stab_rate,
            "y": y[0],
            "y_pertbs": all_y_pertbs,
        }

    else:
        return soft_stab_rate

