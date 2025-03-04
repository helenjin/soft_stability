import torch
import torch.nn as nn
import random
import itertools


def binomial_coefficient(n, k):
    """
    Compute the binomial coefficient (n choose k).
    Args:
        n: Total number of items.
        k: Number of items to choose.

    Returns:
        Binomial coefficient (n choose k).
    """
    if k > n or k < 0:
        return 0

    if not isinstance(n, torch.Tensor):
        n = torch.tensor(n)

    if not isinstance(k, torch.Tensor):
        k = torch.tensor(k)

    return torch.round(torch.exp(
        torch.lgamma(n + 1) - 
        torch.lgamma(k + 1) - 
        torch.lgamma(n - k + 1)
    )).item()


def sample_plus_k_pertb(alpha, k, num_samples=64):
    # Identify the zero indices
    zero_indices = (alpha == 0).nonzero().view(-1)

    if len(zero_indices) < k:
        raise ValueError("Specified perturbation greater than number of zeros")

    # Which of the zero_indices to use.
    which_zeros = torch.stack([torch.randperm(len(zero_indices))[:k] for _ in range(num_samples)])

    # Make a binary mask of the places we'll set hot.
    hot_mask = torch.zeros(num_samples, alpha.size(-1), dtype=bool, device=alpha.device)
    hot_mask.scatter_(1, zero_indices[which_zeros], 1)

    # Mkae a copy of alpha and fill in the entries
    alpha_big = alpha.clone().view(1,-1).repeat(num_samples,1)
    alpha_big.masked_fill_(hot_mask, 1)
    return alpha_big


@torch.no_grad()
def sample_level_k_weights(f, n, k, num_subsets=1024, input_samples=1024):
    """
    Estimate the level-k weights (Fourier coefficients and variance) of a Boolean function f.

    Args:
        f: Callable function that takes a 0/1 tensor of shape (n,) and outputs a tensor of shape (m,).
        n: Number of input variables (dimension of the Boolean hypercube).
        k: The level (degree) of the Fourier weight to sample.
        num_subsets: Number of subsets to sample.
        input_samples: Number of Boolean hypercube inputs to sample.

    Returns:
        An estimate of the level-k variance (Fourier weight).
    """
    assert 0 <= k and k <= n

    device = next(f.parameters()).device if isinstance(f, nn.Module) else "cpu"

    # Randomly sample input points from the Boolean hypercube {0, 1}^n
    inputs = torch.randint(0, 2, (input_samples, n), dtype=torch.float32, device=device)

    # Evaluate the function on the sampled inputs, intentionally do NOT batch some runs
    outputs = torch.cat([f(x.unsqueeze(0)) for x in inputs], dim=0) # (input_samples, m)

    # Randomly sample subsets of size k
    if (k == 0) or (k == n):
        num_subsets = 1
    subsets = [sorted(random.sample(range(n), k)) for _ in range(num_subsets)]

    # Compute Fourier coefficients for sampled subsets
    def compute_fourier_coefficient(S):
        # Compute parity function chi_S(x) for sampled inputs
        chi_S = torch.prod(torch.pow(-1, inputs[:, S]), dim=1) # (input_samples,)
        # Compute Fourier coefficient for S
        return (outputs * chi_S.view(-1,1)).mean(dim=0) # (m,)

    # Sampled Fourier coefficients, (num_subsets, m)
    coefficients = torch.stack([compute_fourier_coefficient(S) for S in subsets])
    return {
        "average_variance": (coefficients ** 2).mean(dim=0), # (m,)
        "average_mass": coefficients.abs().mean(dim=0), # (m,)
        "coefficients": coefficients # (num_subsets, m)
    }

