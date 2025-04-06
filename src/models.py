import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedImageClassifier(nn.Module):
    """
    Wrapper that allows passing both an image and attribution mask to a classifier.
    The mask is upscaled to match the image dimensions before being applied.
    """
    def __init__(
        self,
        base_classifier: nn.Module,
        image_size: tuple[int, int] = (224, 224),
        grid_size: tuple[int, int] = (14, 14)
    ):
        super().__init__()
        self.base_classifier = base_classifier
        self.image_size = image_size
        self.grid_size = grid_size

    def forward(
        self,
        x: torch.FloatTensor,
        alpha: torch.LongTensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass applying attribution mask to input image.

        Args:
            x: Input image tensor of shape (batch_size, channels, height, width)
            alpha: Optional attribution mask of shape (batch_size, grid_height, grid_width)
                  or (batch_size, grid_height * grid_width)

        Returns:
            Model output logits
        """
        bsz, C, H, W = x.shape
        gH, gW = self.grid_size

        # Validate input dimensions
        assert (H, W) == self.image_size

        # Default to all-ones mask if none provided
        if alpha is None:
            alpha = torch.ones(bsz, gH, gW, device=x.device)

        # Upscale mask to image size and apply
        alpha = F.interpolate(alpha.view(bsz, 1, gH, gW).float(), size=(H, W), mode='bilinear')
        out = self.base_classifier(x * alpha)

        # Handle different output formats
        if hasattr(out, "logits"):
            return out.logits
        elif isinstance(out, dict) and "logits" in out:
            return out["logits"]
        return out


class SmoothedImageClassifier(nn.Module):
    """
    Averaged evaluation of an alpha-masked image, where each bit in alpha is kept with 
    probability lambda_.
    """
    def __init__(
        self,
        base_classifier: nn.Module,
        lambda_: float,
        num_samples: int = 64,
        image_size: tuple[int, int] = (224, 224),
        grid_size: tuple[int, int] = (14, 14),
        avg_logits: float = False
    ):
        super().__init__()
        self.masked_image_classifier = MaskedImageClassifier(base_classifier, image_size, grid_size)
        self.mask_dim = grid_size[0] * grid_size[1]
        self.avg_logits = avg_logits

        # If we're close to 1.0, don't bother
        if abs(1 - lambda_) < 1e-4:
            self.lambda_ = 1.0
            self.num_samples = 1
        else:
            self.lambda_ = lambda_
            self.num_samples = num_samples

    def forward(
        self,
        x: torch.FloatTensor,
        alpha: torch.LongTensor | None = None,
    ):
        bsz, C, H, W = x.shape
        N = self.num_samples

        # Add noise if applicable
        keep_mask = torch.rand(bsz, N, self.mask_dim, device=x.device) <= self.lambda_
        if alpha is None:
            alpha = torch.ones(bsz, self.mask_dim, device=x.device)
        alpha = alpha.view(bsz, 1, self.mask_dim) * keep_mask

        # Make copies of x and pass them through the classifier in a batched mode
        xx = x.view(bsz, 1, C, H, W).repeat(1, N, 1, 1, 1)
        y = self.masked_image_classifier(xx.flatten(0,1), alpha.flatten(0,1)).view(bsz, N, -1)
        if self.avg_logits:
            return y.mean(dim=1)
        else:
            return F.one_hot(y.argmax(dim=-1), num_classes=y.size(-1)).float().mean(dim=1)


def discretized_mus_masks(
    dim: int,
    lambda_: float,
    quant: int,
    v_seed: torch.FloatTensor | None = None,
    device: str = "cpu",
    return_all: bool = False,
):
    """
    Sample the discretized MuS noise used by certified classifiers.

    Args:
        dim: The dimension.
        lambda_: The drop prob (and Lipschitz const). Should be a multiple of 1/q.
        quant: The quantization parameter (q).
        v_seed: The seed noise.

    Returns:
        A 0/1-valued mask of shape (q, dim).
    """
    q = quant
    lambda_ = int(lambda_ * q) / q

    if v_seed is None:
        v_seed = torch.randint(0, q, (1, dim), device=device) / q

    s_base = ((torch.arange(q, device=device) + 0.5) / q).view(q,1)
    t = (v_seed + s_base).remainder(1.0) # (q, dim)
    s = (t < lambda_).long()
    if return_all:
        return {"mask": s, "pre_mask": t, "v_seed": v_seed}
    else:
        return s # (q, dim)


class CertifiedImageClassifier(nn.Module):
    """
    A certified variant of Multiplicative Smoothing (MuS) for masked image classifiers.
    
    This class implements a certified defense mechanism for image classifiers by applying
    multiplicative smoothing with discretized noise. It provides both predictions and
    certified robustness radii for the predictions.

    Args:
        base_classifier (nn.Module): The underlying image classifier to be certified
        lambda_ (float): The smoothing parameter (drop probability), should be a multiple of 1/quant
        quant (int, optional): The quantization parameter for discretized noise. Defaults to 64
        image_size (tuple, optional): Expected input image size (H, W). Defaults to (224, 224)
        grid_size (tuple, optional): Grid size for masking (gH, gW). Defaults to (14, 14)

    Returns:
        dict: A dictionary containing:
            - logits (torch.FloatTensor): Classification logits of shape (batch_size, num_classes)
            - cert_rs (torch.FloatTensor): Certified robustness radii of shape (batch_size,)
    """
    def __init__(
        self,
        base_classifier: nn.Module,
        lambda_: float,
        quant: int = 64,
        image_size = (224, 224),
        grid_size = (14, 14)
    ):
        super().__init__()
        self.base_classifier = base_classifier
        self.image_size = image_size
        self.q = quant
        self.lambda_ = int(lambda_ * self.q) / self.q  # Quantize lambda to be a multiple of 1/q
        self.grid_size = grid_size

    def forward(
        self,
        x: torch.FloatTensor,
        alpha: torch.LongTensor | None = None,
    ):
        """
        Forward pass of the certified classifier.

        Args:
            x (torch.FloatTensor): Input images of shape (batch_size, channels, height, width)
            alpha (torch.LongTensor, optional): Attention mask of shape (batch_size, grid_height, grid_width).
                If None, uses all ones. Defaults to None.

        Returns:
            dict: Contains logits and certified robustness radii
        """
        # Shape-check x
        bsz, C, H, W = x.shape
        gH, gW = self.grid_size
        assert (H, W) == self.image_size

        # Make alpha if it does not exist
        if alpha is None:
            alpha = torch.ones(bsz, gH, gW, device=x.device)

        q, mask_dim = self.q, gH * gW
        alpha = alpha.view(bsz, mask_dim)
        all_ys = ()
        for x_, a_ in zip(x, alpha):
            # Generate discretized MuS masks
            mus_masks = discretized_mus_masks(mask_dim, self.lambda_, q, device=x.device) # (q, mask_dim)
            a_masked = a_.view(1, mask_dim) * mus_masks.view(q, mask_dim) # (q, mask_dim)

            # Apply masks to input images
            a_masked = F.interpolate(
                a_masked.view(q, 1, gH, gW).float(),
                size=(H, W),
                mode='bilinear'
            )
            y = self.base_classifier(x_.view(1, C, H, W) * a_masked) # (q, num_classes)

            # Extract logits from various possible output formats
            if hasattr(y, "logits"):
                y = y.logits
            elif isinstance(y, dict) and "logits" in y.keys():
                y = y["logits"]

            # Convert predictions to one-hot and compute average
            y = F.one_hot(y.argmax(dim=-1), num_classes=y.size(-1)) # (q, num_classes)
            avg_y = y.float().mean(dim=0) # (num_classes)
            all_ys += (avg_y,)

        # Stack predictions and compute certified radii
        all_ys = torch.stack(all_ys) # (bsz, num_classes)
        all_ys_desc = all_ys.sort(dim=-1, descending=True).values
        cert_rs = (all_ys_desc[:,0] - all_ys_desc[:,1]) / (2 * self.lambda_)

        return {
            "logits": all_ys, # (bsz, num_classes)
            "cert_rs": cert_rs # (bsz,)
        }


class BinarizedMaskedImageClassifier(torch.nn.Module):
    """
    A wrapper around a masked image classifier that binarizes the input image.
    """
    def __init__(self, masked_image_classifier, image):
        """
        Args:
            masked_image_classifier: The masked image classifier to wrap.
            image: The image to binarize.
        """
        super().__init__()
        self.masked_image_classifier = masked_image_classifier
        self.register_buffer("image", image); assert image.ndim == 3

    def forward(self, alpha):
        """
        Forward pass of the binarized masked image classifier.

        Args:
            alpha: The alpha values of shape (batch_size, grid_height, grid_width)

        Returns:
            The output of the masked image classifier.
        """
        x = self.image.unsqueeze(0).repeat(alpha.size(0), 1, 1, 1)
        return self.masked_image_classifier(x, alpha)


class MaskedTextClassifier(nn.Module):
    """
    A wrapper around a text classifier that supports masked inputs and attention.
    
    This class allows for flexible input handling of either embedded inputs or raw input IDs,
    along with optional attention masks and attribution weights (alpha).
    
    Args:
        base_classifier (nn.Module): The underlying text classifier model that must have
            a get_input_embeddings() method.
            
    Attributes:
        base_classifier (nn.Module): The wrapped classifier model
        embed_fn (nn.Module): The embedding function from the base classifier
        
    Example:
        >>> model = MaskedTextClassifier(base_classifier)
        >>> outputs = model(input_ids=input_ids, attention_mask=attention_mask, alpha=alpha)
    """
    def __init__(
        self,
        base_classifier: nn.Module
    ):
        super().__init__()
        self.base_classifier = base_classifier
        # Assume that the classifier comes with these
        assert hasattr(base_classifier, "get_input_embeddings"), \
            "Classifier must have a get_input_embeddings method"
        
        self.embed_fn = base_classifier.get_input_embeddings()

    def get_input_embeddings(self):
        return self.embed_fn

    def forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        alpha: torch.LongTensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of the masked classifier.
        
        Args:
            inputs_embeds (torch.FloatTensor, optional): Pre-computed embeddings of shape 
                (batch_size, seq_len, hidden_size)
            input_ids (torch.LongTensor, optional): Raw input token IDs of shape 
                (batch_size, seq_len)
            attention_mask (torch.LongTensor, optional): Attention mask of shape 
                (batch_size, seq_len)
            alpha (torch.LongTensor, optional): Attribution weights of shape 
                (batch_size, seq_len)
                
        Returns:
            torch.Tensor: Model logits or raw outputs
            
        Raises:
            AssertionError: If neither inputs_embeds nor input_ids is provided, or if both are provided
        """
        # Exactly one can be present
        assert (input_ids is None) ^ (inputs_embeds is None)

        if inputs_embeds is None:
            device = next(self.parameters()).device
            inputs_embeds = self.embed_fn(input_ids.to(device))

        bsz, L, _ = inputs_embeds.shape

        if attention_mask is None:
            attention_mask = torch.ones(bsz, L, device=inputs_embeds.device).long()

        if alpha is None:
            alpha = torch.ones_like(attention_mask)

        attention_mask = (attention_mask * alpha).long()
        out = self.base_classifier(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        if hasattr(out, "logits"):
            return out.logits
        elif isinstance(out, dict) and "logits" in out.keys():
            return out["logits"]
        else:
            return out


class SmoothedTextClassifier(nn.Module):
    """
    A smoothed version of the masked text classifier that applies random masking to input features.
    
    This class implements multiplicative smoothing where each feature is randomly masked (zeroed)
    with probability 1 - lambda_. The model averages predictions across multiple samples to estimate
    the expected output of the smoothed classifier.
    
    Args:
        base_classifier (nn.Module): The underlying text classifier to smooth
        lambda_ (float): Probability of keeping each feature (smoothing parameter)
        num_samples (int): Number of samples to average over for each input
        avg_logits (bool, optional): Whether to average logits or one-hot predictions. 
            Defaults to False.
    """
    def __init__(
        self,
        base_classifier: nn.Module,
        lambda_: float,
        num_samples: int,
        avg_logits: bool = False
    ):
        super().__init__()
        self.masked_text_classifier = MaskedTextClassifier(base_classifier)
        self.avg_logits = avg_logits

        # Skip smoothing if lambda is very close to 1
        if abs(1 - lambda_) < 1e-4:
            self.lambda_ = 1.0
            self.num_samples = 1
        else:
            self.lambda_ = lambda_
            self.num_samples = num_samples

    def forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        alpha: torch.LongTensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of the smoothed classifier.
        
        Args:
            inputs_embeds (torch.FloatTensor, optional): Pre-computed embeddings
            input_ids (torch.LongTensor, optional): Raw input token IDs
            attention_mask (torch.LongTensor, optional): Attention mask
            alpha (torch.LongTensor, optional): Attribution weights
            
        Returns:
            torch.Tensor: Averaged predictions across samples
            
        Raises:
            AssertionError: If neither inputs_embeds nor input_ids is provided, or if both are provided
        """
        # Validate input arguments
        assert (input_ids is None) ^ (inputs_embeds is None), \
                "Exactly one of input_ids or inputs_embeds must be provided"

        # Get input shape and device
        inputs = input_ids if inputs_embeds is None else inputs_embeds
        bsz, L = inputs.shape[:2]

        # Create default attention mask if none provided
        if attention_mask is None:
            attention_mask = torch.ones(bsz, L, device=inputs.device)

        # Generate random masks for feature dropping
        keep_mask = torch.rand(bsz, self.num_samples, L, device=inputs.device) <= self.lambda_
        
        # Apply attribution weights if provided
        if alpha is None:
            alpha = torch.ones_like(attention_mask)
        alpha = alpha.view(bsz, 1, L) * keep_mask

        # Process inputs based on type
        if input_ids is not None:
            # Handle raw token IDs
            xx = input_ids.view(bsz, 1, L).repeat(1, self.num_samples, 1)
            mm = attention_mask.view(bsz, 1, L).repeat(1, self.num_samples, 1)
            y = self.masked_text_classifier(
                input_ids=xx.flatten(0,1),
                attention_mask=mm.flatten(0,1),
                alpha=alpha.flatten(0,1)
            ).view(bsz, self.num_samples, -1)
        else:
            # Handle pre-computed embeddings
            xx = inputs_embeds.view(bsz, 1, L, -1).repeat(1, self.num_samples, 1, 1)
            mm = attention_mask.view(bsz, 1, L).repeat(1, self.num_samples, 1)
            y = self.masked_text_classifier(
                inputs_embeds=xx.flatten(0,1),
                attention_mask=mm.flatten(0,1),
                alpha=alpha.flatten(0,1)
            ).view(bsz, self.num_samples, -1)

        # Return either averaged logits or averaged one-hot predictions
        if self.avg_logits:
            return y.mean(dim=1)
        else:
            return F.one_hot(y.argmax(dim=-1), num_classes=y.size(-1)).float().mean(dim=1)


class CertifiedTextClassifier(nn.Module):
    """A certified text classifier using multiplicative smoothing with discretized masks.
    
    This class implements a certified text classifier that uses multiplicative smoothing
    with discretized masks to provide certified robustness guarantees. It wraps a base
    classifier and applies discretized multiplicative smoothing during inference.
    
    Args:
        base_classifier (nn.Module): The base text classifier to wrap
        lambda_ (float): The smoothing parameter (probability of keeping each feature)
        quant (int, optional): Number of discrete masks to use. Defaults to 64.
    """
    
    def __init__(
        self,
        base_classifier: nn.Module,
        lambda_: float,
        quant: int = 64,
    ):
        super().__init__()
        self.base_classifier = base_classifier
        self.q = quant
        # Quantize lambda to match discretized masks
        self.lambda_ = int(lambda_ * self.q) / self.q
        
        # Get embedding function from base classifier
        self.embed_fn = base_classifier.get_input_embeddings()

    def forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        alpha: torch.LongTensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the certified classifier.
        
        Args:
            inputs_embeds (torch.FloatTensor | None): Pre-computed embeddings
            input_ids (torch.LongTensor | None): Raw token IDs
            attention_mask (torch.LongTensor | None): Attention mask for padding
            alpha (torch.LongTensor | None): Feature importance weights
            
        Returns:
            dict: Contains:
                - logits: Classification logits (bsz, num_classes)
                - cert_rs: Certified robustness radii (bsz,)
                
        Raises:
            AssertionError: If neither inputs_embeds nor input_ids is provided, or if both are provided
        """
        # Validate input arguments
        assert (input_ids is None) ^ (inputs_embeds is None), \
                "Exactly one of input_ids or inputs_embeds must be provided"

        # Get embeddings if needed
        if inputs_embeds is None:
            inputs_embeds = self.embed_fn(input_ids)

        bsz, L, _ = inputs_embeds.shape

        # Create default attention mask if none provided
        if attention_mask is None:
            attention_mask = torch.ones(bsz, L, device=inputs_embeds.device).long()

        # Create default feature weights if none provided
        if alpha is None:
            alpha = torch.ones_like(attention_mask)

        # Process each example in the batch
        all_ys = ()
        for x_, m_, a_ in zip(inputs_embeds, attention_mask, alpha):
            # Generate discretized multiplicative smoothing masks
            mus_masks = discretized_mus_masks(L, self.lambda_, self.q, device=x_.device)  # (q, L)
            
            # Apply feature importance weights to masks
            a_masked = a_.view(1, L) * mus_masks.view(self.q, L)  # (q, L)
            
            # Apply masks to attention mask and input embeddings
            m_masked = m_.view(1, L) * a_masked  # (q, L)
            x_big = x_.view(1, L, -1).repeat(self.q, 1, 1)

            # Get predictions from base classifier
            y = self.base_classifier(inputs_embeds=x_big, attention_mask=m_masked)

            # Extract logits from various possible return types
            if hasattr(y, "logits"):
                y = y.logits
            elif isinstance(y, dict) and "logits" in y.keys():
                y = y["logits"]

            # Convert to one-hot predictions and average across masks
            y = F.one_hot(y.argmax(dim=-1), num_classes=y.size(-1))  # (q, num_classes)
            avg_y = y.float().mean(dim=0)  # (num_classes,)
            all_ys += (avg_y,)

        # Stack predictions and compute certified radii
        all_ys = torch.stack(all_ys)  # (bsz, num_classes)
        all_ys_desc = all_ys.sort(dim=-1, descending=True).values
        cert_rs = (all_ys_desc[:,0] - all_ys_desc[:,1]) / (2 * self.lambda_)

        return {
            "logits": all_ys,    # (bsz, num_classes)
            "cert_rs": cert_rs   # (bsz,)
        }


class BinarizedMaskedTextClassifier(torch.nn.Module):
    """
    A wrapper around a masked text classifier that binarizes the input text.
    """
    def __init__(self, masked_text_classifier, input_ids):
        """
        Args:
            masked_text_classifier: The masked text classifier to wrap.
            input_ids: The input IDs to binarize.
        """
        super().__init__()
        self.masked_text_classifier = masked_text_classifier
        self.register_buffer("input_ids", input_ids.view(-1))

    def forward(self, alpha):
        """
        Forward pass of the binarized masked text classifier.

        Args:
            alpha: The alpha values of shape (batch_size, seq_len)

        Returns:
            The output of the masked text classifier.
        """
        input_ids = self.input_ids.unsqueeze(0).repeat(alpha.size(0), 1)
        return self.masked_text_classifier(input_ids=input_ids, attention_mask=alpha)