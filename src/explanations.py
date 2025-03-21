import torch

import sys
# sys.path.append('/shared_data0/helenjin/exlib/src')
sys.path.append('/shared_data0/weiqiuy/exlib/src')
import exlib

from exlib.explainers import LimeImageCls, ShapImageCls, IntGradImageCls, MfabaImageCls
from exlib.explainers.lime import LimeTextCls
from exlib.explainers.shap import ShapTextCls
from exlib.explainers.intgrad import IntGradTextCls
from exlib.explainers.mfaba import MfabaTextCls
from exlib.explainers.common import patch_segmenter


"""
For image explanations, we make the default assumption that:
    An image tensor has shape (C, 224, 224)
    The number of patches is 196, which is a 14x14 grid
    num_patches = 196
"""


def get_lime_for_image(
    model,
    image: torch.FloatTensor,
    pred = None,
    patch_segmenter = patch_segmenter,
    num_patches: int = 196,
    num_samples: int = 1000,
    top_k_frac: float = 0.25,
    return_verbose: bool = False
):
    device = next(model.parameters()).device
    X = image.unsqueeze(0).to(device)
    if pred == None:
        logits = model(X)
        pred = torch.argmax(logits, dim=-1)

    # call LIME
    eik = {
        "segmentation_fn": patch_segmenter,
        "top_labels": 1000, 
        "hide_color": 0, 
        "num_samples": num_samples
    }
    gimk = {
        "positive_only": False
    }
    liek = {
        "random_state": 1
    }

    explainer = LimeImageCls(model, 
                             explain_instance_kwargs=eik, 
                             get_image_and_mask_kwargs=gimk, 
                             LimeImageExplainerKwargs=liek)
    
    expln = explainer(X, pred)
    num_patches_one_axis = int(num_patches**0.5)
    expln_alpha = get_alpha_from_img_attrs(expln.attributions, image_size=image.shape[1], num_patches_one_axis=num_patches_one_axis, masked=True, top_k_frac=top_k_frac)
    if return_verbose:
        return expln_alpha, expln, expln.attributions
    return expln_alpha
    # return (torch.rand(num_patches) < top_k_frac).long()


def get_shap_for_image(
    model,
    image: torch.FloatTensor,
    pred = None,
    num_patches: int = 196,
    num_samples: int = 1000,
    top_k_frac: float = 0.25,
    return_verbose: bool = False
):
    device = next(model.parameters()).device
    X = image.unsqueeze(0).to(device)
    if pred == None:
        logits = model(X)
        pred = torch.argmax(logits, dim=-1)

    # call SHAP
    sek = {'max_evals': num_samples}
    explainer = ShapImageCls(model, shap_explainer_kwargs=sek)

    expln = explainer(X, pred)
    num_patches_one_axis = int(num_patches**0.5)
    expln_alpha = get_alpha_from_img_attrs(expln.attributions, image_size=image.shape[1], num_patches_one_axis=num_patches_one_axis, masked=True, top_k_frac=top_k_frac)
    if return_verbose:
        return expln_alpha, expln, expln.attributions
    return expln_alpha
    # return (torch.rand(num_patches) < top_k_frac).long()


def get_intgrad_for_image(
    model,
    image: torch.FloatTensor,
    pred = None,
    num_patches: int = 196,
    num_samples: int = 1000,
    top_k_frac: float = 0.25,
    return_verbose: bool = False
):
    device = next(model.parameters()).device
    X = image.unsqueeze(0).to(device)
    if pred == None:
        logits = model(X)
        pred = torch.argmax(logits, dim=-1)
        
    # call IntGrad
    explainer = IntGradImageCls(model, num_steps=num_samples)
    
    expln = explainer(X, pred)
    num_patches_one_axis = int(num_patches**0.5)
    expln_alpha = get_alpha_from_img_attrs(expln.attributions, image_size=image.shape[1], num_patches_one_axis=num_patches_one_axis, masked=True, top_k_frac=top_k_frac)
    if return_verbose:
        return expln_alpha, expln, expln.attributions
    return expln_alpha
    # return (torch.rand(num_patches) < top_k_frac).long()


def get_mfaba_for_image(
    model,
    image: torch.FloatTensor,
    pred = None,
    num_patches: int = 196,
    top_k_frac: float = 0.25,
    return_verbose: bool = False
):
    device = next(model.parameters()).device
    X = image.unsqueeze(0).to(device)
    if pred == None:
        logits = model(X)
        pred = torch.argmax(logits, dim=-1)
        
    # call MfabaGrad
    explainer = MfabaImageCls(model)
    
    expln = explainer(X, pred)
    num_patches_one_axis = int(num_patches**0.5)
    expln_alpha = get_alpha_from_img_attrs(expln.attributions, image_size=image.shape[1], num_patches_one_axis=num_patches_one_axis, masked=True, top_k_frac=top_k_frac)
    if return_verbose:
        return expln_alpha, expln, expln.attributions
    return expln_alpha
    # return (torch.rand(num_patches) < top_k_frac).long()


    
def get_random_for_image(expln_shape = (1, 14, 14), top_k_frac=0.25):
    total_feats = expln_shape[1] * expln_shape[2]
    expln_size = int(total_feats * top_k_frac)
    index = torch.randperm(total_feats)[:expln_size]
    random_expln = torch.zeros(total_feats)
    random_expln.index_fill_(0, index, 1)
    return random_expln.view(expln_shape)



def get_alpha_from_img_attrs(attrs, image_size=224, num_patches_one_axis=14, masked=True, top_k_frac=0.25):
    """
    Convert input tensor into a patch version by averaging over patches.
    
    Parameters:
        attrs (torch.Tensor): Input tensor of shape (N, C, image_size, image_size).
        image_size (int): Size of the input image (assumed square).
        num_patches_one_axis (int): Number of patches along one dimension.
        masked (bool): If True, return a boolean mask based on top_k_frac.
        top_k_frac (float): Fraction of top elements to include in the mask.
        
    Returns:
        torch.Tensor: Patch version of the input tensor of shape (N, num_patches_one_axis, num_patches_one_axis).
        (If masked=True, return the boolean mask where the threshold is top_k_frac).
    """
    # Calculate the approximate patch size and the leftover regions
    patch_size = image_size // num_patches_one_axis
    remainder = image_size % num_patches_one_axis

    # Generate dynamic patch sizes with adjusted padding for the last patch
    patch_sizes = [patch_size + 1 if i < remainder else patch_size for i in range(num_patches_one_axis)]

    # Cumulative sum to identify patch boundaries
    boundaries = [0] + torch.cumsum(torch.tensor(patch_sizes), 0).tolist()

    N, C, H, W = attrs.shape
    if H != image_size or W != image_size:
        raise ValueError(f"Input tensor height and width must match the given image_size ({image_size}).")
    
    # Aggregate values into patches
    patch_tensor = torch.zeros((N, num_patches_one_axis, num_patches_one_axis), device=attrs.device)
    for i in range(num_patches_one_axis):
        for j in range(num_patches_one_axis):
            patch = attrs[
                :, :, boundaries[i]:boundaries[i + 1], boundaries[j]:boundaries[j + 1]
            ]
            patch_tensor[:, i, j] = patch.mean(dim=(-1, -2, -3))

    result = patch_tensor
    if masked:
        threshold_num = int(top_k_frac * len(result[0].flatten()))
        result_masked = []
        for i in range(N):
            topk_indices = torch.topk(result[i].flatten(), threshold_num).indices
            result_masked_i = torch.zeros_like(result[i].flatten())
            result_masked_i[topk_indices] = 1
            result_masked.append(result_masked_i)
        result_masked = torch.stack(result_masked)
        return result_masked.view(result.shape).to(attrs.device)
    else:
        return result


def get_lime_for_text(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    tokenizer,
    pred = None,
    num_samples: int = 1000,
    num_features: int = 150,
    top_labels: int = 4,
    top_k_frac: float = 0.25,
    return_verbose: bool = False
):
    def split_expression(x):
        tokens = x.split()
        return tokens
    
    if pred == None:
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(logits, dim=-1)

    # call LIME
    eik = {
    "top_labels": top_labels, 
    "num_samples": num_samples,
    "num_features": num_features
    }
    ltek = {
    "mask_string": "[MASK]",
    "split_expression": split_expression,
    "feature_selection": 'none'
    }

    explainer = LimeTextCls(model, tokenizer,
                            LimeTextExplainerKwargs=ltek,
                            explain_instance_kwargs=eik).cuda()
    expln = explainer(input_ids, pred)
    attrs = expln.attributions
    expln_alpha = get_alpha_from_text_attrs(attrs, top_k_frac=top_k_frac)
    
    if return_verbose:
        return expln_alpha, expln, attrs
    return expln_alpha


def get_shap_for_text(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    tokenizer,
    pred = None,
    top_k_frac: float = 0.25,
    return_verbose: bool = False
):
    if pred == None:
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(logits, dim=-1)

    # call SHAP
    explainer = ShapTextCls(model, tokenizer, pad_value=1, special_tokens=['<s>', '</s>'])
    
    expln = explainer(input_ids, pred)
    attrs = expln.attributions #.squeeze()
    expln_alpha = get_alpha_from_text_attrs(attrs, top_k_frac=top_k_frac)
    
    if return_verbose:
        return expln_alpha, expln, attrs
    return expln_alpha
    
    

def get_intgrad_for_text(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    pred = None,
    top_k_frac: float = 0.25,
    return_verbose: bool = False
):
    if pred == None:
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(logits, dim=-1)

    # call IntGrad
    projection_layer = model.base_classifier.roberta.embeddings.word_embeddings
    explainer = IntGradTextCls(model, projection_layer=projection_layer) 
        
    expln = explainer(input_ids.to(model.device), pred)
    attrs = expln.attributions
    expln_alpha = get_alpha_from_text_attrs(attrs, top_k_frac=top_k_frac)
    
    if return_verbose:
        return expln_alpha, expln, attrs
    return expln_alpha


def get_mfaba_for_text(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    pred = None,
    top_k_frac: float = 0.25,
    return_verbose: bool = False
):
    if pred == None:
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(logits, dim=-1)

    # call MFABA
    projection_layer = model.base_classifier.roberta.embeddings.word_embeddings
    explainer = MfabaTextCls(model, projection_layer=projection_layer) 
        
    expln = explainer(input_ids.to(model.device), pred)
    attrs = expln.attributions
    expln_alpha = get_alpha_from_text_attrs(attrs, top_k_frac=top_k_frac)
    
    if return_verbose:
        return expln_alpha, expln, attrs
    return expln_alpha


def get_random_for_text(expln_shape, top_k_frac=0.25):
    total_feats = expln_shape[1]
    expln_size = int(total_feats * top_k_frac)
    index = torch.randperm(total_feats)[:expln_size]
    random_expln = torch.zeros(total_feats)
    random_expln.index_fill_(0, index, 1)
    return random_expln.view(expln_shape)

    
def get_alpha_from_text_attrs(attrs, top_k_frac=0.25):
    """
    Create a binary mask for text attributes by thresholding to the top_k_frac.

    Parameters:
        attrs (torch.Tensor): Input tensor of shape (bsz, l) with real numbers.
        top_k_frac (float): Fraction of top elements to retain in the mask.

    Returns:
        torch.Tensor: Binary mask of shape (bsz, l).
    """
    bsz, l = attrs.shape
    threshold_num = int(top_k_frac * l)  # Number of top elements to retain

    result_masked = []
    for i in range(bsz):
        topk_indices = torch.topk(attrs[i], threshold_num).indices
        result_masked_i = torch.zeros_like(attrs[i])
        result_masked_i[topk_indices] = 1
        result_masked.append(result_masked_i)

    return torch.stack(result_masked).to(attrs.device)
