import torch
from transformers import AutoTokenizer, ViTImageProcessor
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
import os
import linecache

# Global ViT image processor for consistent image preprocessing
vit_image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


def load_image_from_path(image_path: str, resize_to: tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Load and preprocess a single image from a file path.
    
    Args:
        image_path: Path to the image file
        resize_to: Target size for the image (height, width)
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    image = PIL.Image.open(image_path).convert("RGB")
    ret = vit_image_processor(
        image,
        do_resize=True,
        size={"height": resize_to[0], "width": resize_to[1]},
        return_tensors="pt"
    )
    image_pt = ret["pixel_values"]
    return image_pt.squeeze(0) if image_pt.ndim == 4 else image_pt


def load_images_from_directory(
    directory_path: str,
    resize_to: tuple[int, int] = (224, 224),
    max_amount: int | None = None
) -> torch.Tensor:
    """Load and preprocess all images from a directory.
    
    Args:
        directory_path: Path to directory containing images
        resize_to: Target size for the images (height, width)
        max_amount: Maximum number of images to load
        
    Returns:
        torch.Tensor: Stacked tensor of preprocessed images
    """
    image_paths = sorted([
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    image_tensors: list[torch.Tensor] = []
    for path in image_paths:
        try:
            image_tensors.append(load_image_from_path(path, resize_to=resize_to))
        except Exception as e:
            print(f"Failed to load {path}: {str(e)}")

        if max_amount is not None and len(image_tensors) >= max_amount:
            break

    return torch.stack(image_tensors)


class ImageDataset(Dataset):
    """Dataset for loading and preprocessing images from a directory.
    
    Supports both ViT preprocessing and basic torchvision transforms.
    
    Args:
        image_folder: Path to directory containing images
        image_size: Target size for images (height, width)
        run_preprocessor: Whether to use ViT preprocessing or basic transforms
    """
    def __init__(
        self,
        image_folder: str,
        image_size: tuple[int, int] = (224, 224),
        use_preprocessor: bool = True
    ):
        self.image_folder = image_folder
        self.image_size = image_size
        self.image_paths = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        self.preprocessor = vit_image_processor if use_preprocessor else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_pil = PIL.Image.open(self.image_paths[idx]).convert("RGB")

        if self.preprocessor is not None:
            image_pt = self.preprocessor(
                image_pil,
                do_resize=True,
                size={"height": self.image_size[0], "width": self.image_size[1]},
                return_tensors="pt"
            )["pixel_values"].squeeze(0)
        else:
            image_pt = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])(image_pil)

        return image_pt


class TweetDataset(Dataset):
    """Dataset for loading and preprocessing tweets with labels.
    
    Supports various Twitter-specific NLP tasks using RoBERTa tokenization.
    
    Args:
        text_path: Path to file containing tweet texts
        labels_path: Path to file containing tweet labels
        task: Type of NLP task (emoji, emotion, hate, irony, offensive, sentiment, stance)
    """
    def __init__(self, text_path: str, labels_path: str, task: str = 'emotion'):
        valid_tasks = ['emoji', 'emotion', 'hate', 'irony', 'offensive', 'sentiment', 'stance']
        if task not in valid_tasks:
            raise ValueError(f"Task must be one of {valid_tasks}")
            
        self.text_path = text_path
        self.labels_path = labels_path
        self.tokenizer = AutoTokenizer.from_pretrained(f"cardiffnlp/twitter-roberta-base-{task}")
        
        with open(text_path, "r") as f:
            self.num_data = len(f.readlines())
        with open(labels_path, "r") as f:
            self.labels = [int(l.strip()) for l in f]

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int) -> tuple[dict, int]:
        def preprocess(text: str) -> str:
            """Preprocess tweet text by normalizing mentions and URLs."""
            new_text = []
            for t in text.split(" "):
                t = '@user' if t.startswith('@') and len(t) > 1 else t
                t = 'http' if t.startswith('http') else t
                new_text.append(t)
            return " ".join(new_text)

        text = linecache.getline(self.text_path, idx + 1)
        text = preprocess(text)
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            max_length=512,
            truncation=True
        )
        return inputs, self.labels[idx]


class TweetEvalDataset(Dataset):
    """Dataset for loading and preprocessing tweets with labels.
    
    Supports various Twitter-specific NLP tasks using RoBERTa tokenization.
    
    Args:
        task: Type of NLP task (emoji, emotion, hate, irony, offensive, sentiment, stance)
        dataset_dir: Path to directory containing the dataset
    """
    def __init__(self, task: str, datasets_dir: str):
        assert task in ['emoji', 'emotion', 'hate', 'irony', 'offensive', 'sentiment', 'stance']
        self.task = task
        self.text_path = os.path.join(datasets_dir, task, "val_text.txt")
        self.labels_path = os.path.join(datasets_dir, task, "val_labels.txt")
        
        with open(self.text_path, "r") as f:
            self.num_data = len(f.readlines())

        with open(self.labels_path, "r") as f:
            self.labels = [int(l.strip()) for l in f]

        self.tokenizer = AutoTokenizer.from_pretrained(f"cardiffnlp/twitter-roberta-base-{task}")

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx: int) -> tuple[dict, int]:
        def preprocess(text: str) -> str:
            """Preprocess tweet text by normalizing mentions and URLs."""
            new_text = []
            for t in text.split(" "):
                t = '@user' if t.startswith('@') and len(t) > 1 else t
                t = 'http' if t.startswith('http') else t
                new_text.append(t)
            return " ".join(new_text)

        text = linecache.getline(self.text_path, idx + 1)
        text = preprocess(text)
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            max_length=512,
            truncation=True
        )
        return inputs, self.labels[idx]
