import torch
import torch.nn as nn
import random


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
    
    # Convert inputs to tensors if needed
    n = torch.tensor(n) if not isinstance(n, torch.Tensor) else n
    k = torch.tensor(k) if not isinstance(k, torch.Tensor) else k
    
    # Compute using log gamma for numerical stability
    result = torch.exp(
        torch.lgamma(n + 1) -
        torch.lgamma(k + 1) -
        torch.lgamma(n - k + 1)
    )
    
    return torch.round(result).item()


@torch.no_grad()
def sample_level_k_weights(f, n, k, num_subsets=1024, input_samples=1024):
    """
    Estimate the level-k weights (Fourier coefficients and variance) of a Boolean function f.

    Args:
        f: Callable function that takes a 0/1 tensor of shape (n,) and outputs a tensor of 
           shape (m,).
        n: Number of input variables (dimension of the Boolean hypercube).
        k: The level (degree) of the Fourier weight to sample.
        num_subsets: Number of subsets to sample.
        input_samples: Number of Boolean hypercube inputs to sample.

    Returns:
        Dictionary containing:
            - average_variance: Mean squared Fourier coefficients at level k
            - average_mass: Mean absolute Fourier coefficients at level k
            - coefficients: All sampled Fourier coefficients
    """
    assert 0 <= k <= n

    device = next(f.parameters()).device if isinstance(f, nn.Module) else "cpu"

    # Sample random inputs from {0,1}^n
    inputs = torch.randint(0, 2, (input_samples, n), dtype=torch.float32, device=device)
    
    # Evaluate function on inputs
    outputs = torch.cat([f(x.unsqueeze(0)) for x in inputs], dim=0)

    # For k=0 or k=n, only one subset exists
    num_subsets = 1 if (k == 0 or k == n) else num_subsets
    
    # Sample random k-sized subsets
    subsets = [sorted(random.sample(range(n), k)) for _ in range(num_subsets)]

    # Compute Fourier coefficients for each subset
    coefficients = []
    for S in subsets:
        # Compute parity function chi_S(x) = prod_{i in S} (-1)^x_i
        chi_S = torch.prod(torch.pow(-1, inputs[:, S]), dim=1)
        # Take mean of f(x) * chi_S(x)
        coeff = (outputs * chi_S.view(-1,1)).mean(dim=0)
        coefficients.append(coeff)
    
    coefficients = torch.stack(coefficients)
    # coefficients shape: (num_subsets, m) where m is output dimension
    # average_variance shape: (m,)
    # average_mass shape: (m,)
    return {
        "average_variance": (coefficients ** 2).mean(dim=0),  # Mean squared coefficients
        "average_mass": coefficients.abs().mean(dim=0),  # Mean absolute coefficients
        "coefficients": coefficients  # All sampled coefficients
    }
