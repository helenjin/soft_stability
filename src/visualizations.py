import torch
from itertools import combinations
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models import discretized_mus_masks

import random
import numpy as np
import torch
import os

def add_k_ones_enumerate(tensor, k):
    """
    Adds k ones to the tensor where it is 0, enumerating all possible ways.

    Parameters:
        tensor (torch.Tensor): The input tensor with 0s and 1s.
        k (int): The number of ones to add.

    Returns:
        list of torch.Tensor: A list of tensors, each representing a unique way to add k ones.
    """
    # Flatten the tensor to work with indices
    flat_tensor = tensor.flatten()
    zero_indices = (flat_tensor == 0).nonzero(as_tuple=True)[0]  # Indices of 0s
    
    # Check if there are enough zeros to add k ones
    if len(zero_indices) < k:
        raise ValueError("Not enough zeros in the tensor to add k ones.")
    
    # Generate all combinations of indices to add ones
    combinations_of_indices = list(combinations(zero_indices.tolist(), k))
    
    # Create new tensors with ones added at the chosen positions
    results = []
    for indices in combinations_of_indices:
        new_tensor = flat_tensor.clone()
        for idx in indices:
            new_tensor[idx] = 1
        results.append(new_tensor.view(tensor.shape))  # Reshape back to original shape

    return results


def vis_masks(image: torch.Tensor, alpha: torch.Tensor, ax=plt, save_path=None, 
                   alpha_original: torch.Tensor = None, linewidths=5):
    """
    Visualize an image with an overlaid transparency mask and optional contour for difference.
    
    Args:
        image (torch.Tensor): Shape (1, 3, 224, 224), normalized in range [-1, 1].
        alpha (torch.Tensor): Shape (1, num_patches, num_patches).
        ax: Matplotlib axis or plt itself.
        save_path (str, optional): If provided, saves the figure to this path.
        alpha_original (torch.Tensor, optional): Original alpha for computing the difference.
    """
    image = torch.ones_like(image)
    mask = F.interpolate(alpha[:, None], size=(224, 224)).cpu()[0, 0]
    
    if alpha_original is not None:
        mask_original = F.interpolate(alpha_original[:, None], size=(224, 224)).cpu()[0, 0]
        diff_mask = (mask - mask_original).numpy()
    else:
        diff_mask = None
    
    if ax == plt:
        plt.figure()
    
    ax.imshow(((image[0] + 1) / 2 * mask).cpu().permute(1, 2, 0))
    ax.axis('off')
    
    # print('diff_mask', diff_mask)
    # print('mask_original', mask_original)
    
    # Draw contour if alpha_original is provided
    if diff_mask is not None:
        contour_levels = np.linspace(diff_mask.min(), diff_mask.max(), 10)
        ax.contour(diff_mask, levels=[0], colors='yellow', linewidths=linewidths)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()
        
        
def vis_masked_img(image: torch.Tensor, alpha: torch.Tensor, ax=plt, save_path=None, alpha_original: torch.Tensor = None, linewidths=5):
    """
    Visualize an image with an overlaid transparency mask and optional contour for difference.
    
    Args:
        image (torch.Tensor): Shape (1, 3, 224, 224), normalized in range [-1, 1].
        alpha (torch.Tensor): Shape (1, num_patches, num_patches).
        ax: Matplotlib axis or plt itself.
        save_path (str, optional): If provided, saves the figure to this path.
        alpha_original (torch.Tensor, optional): Original alpha for computing the difference.
    """
    mask = F.interpolate(alpha[:, None], size=(224, 224)).cpu()[0, 0]
    
    if alpha_original is not None:
        mask_original = F.interpolate(alpha_original[:, None], size=(224, 224)).cpu()[0, 0]
        diff_mask = (mask - mask_original).numpy()
    else:
        diff_mask = None
    
    if ax == plt:
        plt.figure()
    
    ax.imshow(((image[0] + 1) / 2 * mask).cpu().permute(1, 2, 0))
    ax.axis('off')
    
    # print('diff_mask', diff_mask)
    # print('mask_original', mask_original)
    
    # Draw contour if alpha_original is provided
    if diff_mask is not None:
        contour_levels = np.linspace(diff_mask.min(), diff_mask.max(), 10)
        ax.contour(diff_mask, levels=[0], colors='yellow', linewidths=linewidths)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

# plt.imshow()

# Assuming `raw_vit` is your model and `wrapped_vit` is the function wrapping the model
def get_top_k_predictions(wrapped_vit, image, raw_vit, k=5):
    """
    Get the top k predictions from the model with both logits and probabilities.

    Parameters:
        wrapped_vit (function): A wrapper around the Vision Transformer model.
        image (torch.Tensor): The input image tensor.
        raw_vit (object): The model configuration object with id2label mapping.
        k (int): The number of top predictions to return.

    Returns:
        List[Tuple[int, str, float, float]]: Top k predictions as a list of 
                                             (label_id, label_name, logit, probability).
    """
    # Pass the image through the model
    logits = wrapped_vit(image.cuda())  # Shape: [batch_size, num_classes]

    # Convert logits to probabilities using softmax
    probabilities = F.softmax(logits, dim=-1)

    # Get the top k logits, probabilities, and their indices
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    top_k_probs = probabilities.gather(1, top_k_indices)

    # Map the indices to labels, logits, and probabilities
    top_k_predictions = [
        (
            idx.item(),
            raw_vit.config.id2label[idx.item()],
            logit.item(),
            prob.item()
        )
        for idx, logit, prob in zip(top_k_indices[0], top_k_logits[0], top_k_probs[0])
    ]

    return top_k_predictions


def get_labels_containing_keyword(id2label, keyword="russian"):
    """
    Get all labels that contain a specific keyword.

    Parameters:
        id2label (dict): A dictionary mapping label IDs to label names.
        keyword (str): The keyword to search for (case-insensitive).

    Returns:
        List[Tuple[int, str]]: List of (label_id, label_name) for matching labels.
    """
    keyword = keyword.lower()  # Make the search case-insensitive
    matching_labels = [
        (label_id, label_name)
        for label_id, label_name in id2label.items()
        if keyword in label_name.lower()
    ]
    return matching_labels


def set_random_seed(seed):
    """
    Set all random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
    
    Parameters:
        seed (int): The seed value to set.
    """
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy random
    torch.manual_seed(seed)  # PyTorch CPU random seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU random seed
    torch.cuda.manual_seed_all(seed)  # PyTorch all GPU random seeds
    
    # For deterministic behavior in PyTorch (if required)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedImageClassifierNew(nn.Module):
    """
    Allows one to pass both an image x and attribution alpha to the classifier.
    """
    def __init__(
        self,
        base_model: nn.Module,
        image_size: tuple[int, int] = (224, 224),
        grid_size: tuple[int, int] = (14, 14)
    ):
        super().__init__()
        self.base_model = base_model
        self.image_size = image_size
        self.grid_size = grid_size
        self.scale_factor = (image_size[0] // grid_size[0], image_size[1] // grid_size[1])
        self.device = base_model.device

    def forward(
        self,
        x: torch.FloatTensor,
        alpha: torch.LongTensor | None = None
    ):
        """
        x: (bsz, C, H, W)
        alpha: (bsz, *grid_size) or (bsz, grid_size[0] * grid_size[1]).
        """
        # Shape-check x
        bsz, C, H, W = x.shape
        # assert (H, W) == self.image_size

        gH, gW = self.grid_size
        # assert (H, W) == (gH * self.scale_factor[0], gW * self.scale_factor[1])

        # Make alpha if it does not exist
        if alpha is None:
            alpha = torch.ones(bsz, gH, gW, device=x.device)

        # Mask x (bsz, C, H, W) with an up-scaled alpha (bsz, 1, H, W)
        alpha = F.interpolate(alpha.view(bsz, 1, gH, gW).float(), size=(H, W))
        out = self.base_model(x * alpha)

        if hasattr(out, "logits"):
            return out.logits
        elif isinstance(out, dict) and "logits" in out.keys():
            return out["logits"]
        else:
            return out
        

class CertifiedMuSImageClassifierNew(nn.Module):
    """
    The certified variant of MuS for masked image classifiers, with optional masking
    """
    def __init__(
        self,
        base_model: nn.Module,
        lambda_: float,
        quant: int = 64,
        image_size = (224, 224),
        grid_size = (14, 14)
    ):
        super().__init__()
        self.base_model = base_model
        self.image_size = image_size
        self.q = quant
        self.lambda_ = int(lambda_ * self.q) / self.q
        self.grid_size = grid_size
        self.scale_factor = (image_size[0] // grid_size[0], image_size[1] // grid_size[1])

    def forward(
        self,
        x: torch.FloatTensor,
        alpha: torch.LongTensor | None = None,
        return_all: bool = False
    ):
        # Shape-check x
        bsz, C, H, W = x.shape
        # assert (H, W) == self.image_size

        gH, gW = self.grid_size
        # assert (H, W) == (gH * self.scale_factor[0], gW * self.scale_factor[1])

        # Make alpha if it does not exist
        if alpha is None:
            alpha = torch.ones(bsz, gH, gW, device=x.device)

        q, mask_dim = self.q, gH * gW
        alpha = alpha.view(bsz, mask_dim)
        all_ys = ()
        for x_, a_ in zip(x, alpha):
            mus_masks = discretized_mus_masks(mask_dim, self.lambda_, q, device=x.device) # (q, mask_dim)
            a_masked = a_.view(1, mask_dim) * mus_masks.view(q, mask_dim) # (q, mask_dim)
            
            # Multiply x (C, H, W) with an up-scaled a_masked (q, 1, H, W)
            a_masked = F.interpolate(a_masked.view(q, 1, gH, gW).float(), 
                                     size=(H, W))
                                     # scale_factor=self.scale_factor)
            y = self.base_model(x_.view(1, C, H, W) * a_masked) # (q, num_classes)

            # Extract logits as necessary
            if hasattr(y, "logits"):
                y = y.logits
            elif isinstance(y, dict) and "logits" in y.keys():
                y = y["logits"]

            # Convert to one-hot and vote
            y = F.one_hot(y.argmax(dim=-1), num_classes=y.size(-1)) # (q, num_classes)
            avg_y = y.float().mean(dim=0) # (num_classes)
            all_ys += (avg_y,)

        all_ys = torch.stack(all_ys) # (bsz, num_classes)
        all_ys_desc = all_ys.sort(dim=-1, descending=True).values
        cert_rs = (all_ys_desc[:,0] - all_ys_desc[:,1]) / (2 * self.lambda_)

        return {
            "logits": all_ys, # (bsz, num_classes) 
            "cert_rs": cert_rs # (bsz,)
        }
    
    
if __name__ == "__main__":
    # Example usage
    shap_expln_alpha = torch.tensor([[[1., 0., 0.],
                                       [1., 1., 0.],
                                       [1., 0., 0.]]], device='cuda:0')
    k = 2
    results = add_k_ones_enumerate(shap_expln_alpha, k)

    # Print results
    for i, result in enumerate(results):
        print(f"Result {i + 1}:\n{result}")
