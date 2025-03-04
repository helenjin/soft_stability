import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer, ViTImageProcessor
import torchvision.transforms as tvtfs

from transformers import BatchEncoding
from torch.utils.data import Dataset

import PIL
import os
import linecache

vit_image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")


def load_image_from_path(image_path: str, resize_to=(224,224)):
    image = PIL.Image.open(image_path).convert("RGB")
    ret = vit_image_processor(
        image,
        do_resize = True,
        size = {"height": resize_to[0], "width": resize_to[1]},
        return_tensors = "pt"
    )
    image_pt = ret["pixel_values"]
    return image_pt.squeeze(0) if image_pt.ndim == 4 else image_pt


def load_images_from_directory(directory_path: str, resize_to=(224,224), max_amount: int | None = None):
    image_paths = sorted([
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    image_tensors = []
    for path in image_paths:
        try:
            image_tensors.append(load_image_from_path(path, resize_to=resize_to))
        except:
            print(f"Failed to load: {path}")

        if max_amount is not None:
            if len(image_tensors) > max_load:
                break

    return torch.stack(image_tensors)


# Dataset class for tweets
class TweetDataset(Dataset):
  def __init__(self, text_path, labels_path, task='emotion'):
    assert task in ['emoji','emotion', 'hate', 'irony', 'offensive', 'sentiment', 'stance']
    self.text_path = text_path
    self.labels_path = labels_path
    self.num_data = 0
    self.tokenizer = AutoTokenizer.from_pretrained(f"cardiffnlp/twitter-roberta-base-{task}")
    with open(text_path, "r") as f:
      self.num_data = len(f.readlines())

    with open(labels_path, "r") as f:
      self.labels = [int(l.strip()) for l in f]

  def __len__(self):
    return self.num_data

  def __getitem__(self, idx):
    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    text = linecache.getline(self.text_path, idx+1)
    text = preprocess(text)
    inputs = self.tokenizer(text, return_tensors='pt', padding=True, max_length=512,
                           truncation=True)
    return inputs, self.labels[idx]


