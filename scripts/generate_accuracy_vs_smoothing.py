import os
import sys
import json
import argparse
from tqdm import tqdm
import torch
import torchvision.models as tvm
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, RobertaForSequenceClassification

sys.path.append("../src")
from models import SmoothMaskedImageClassifier, SmoothMaskedTextClassifier
from data_utils import ImageNetSubset, TweetEvalDataset


IMAGENET_SAMPLES_DIR = "/home/antonxue/foo/data/imagenet_samples"
TWEETEVAL_DIR = "/home/antonxue/foo/data/tweeteval/datasets"


def load_model_dataset(
    model_name: str,
    dataset_name: str,
    lambda_: float,
    num_samples: int,
):
    if model_name == "vit":
        model = SmoothMaskedImageClassifier(
            ViTForImageClassification.from_pretrained("google/vit-base-patch16-224"),
            lambda_=lambda_,
            num_samples=num_samples,
        )
    elif model_name == "resnet18":
        model = SmoothMaskedImageClassifier(
            tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1),
            lambda_=lambda_,
            num_samples=num_samples,
        )
    elif model_name == "resnet50":
        model = SmoothMaskedImageClassifier(
            tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1),
            lambda_=lambda_,
            num_samples=num_samples,
        )
    elif model_name == "roberta":
        _, task = dataset_name.split("_")
        model = SmoothMaskedTextClassifier(
            RobertaForSequenceClassification.from_pretrained(f"cardiffnlp/roberta-base-{task}"),
            lambda_=lambda_,
            num_samples=num_samples,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")

    if dataset_name.startswith("imagenet") and dataset_name.endswith("per_class"):
        dataset = ImageNetSubset(
            IMAGENET_SAMPLES_DIR + "/" + dataset_name,
            image_size=(224, 224),
            use_preprocessor=True
        )

    elif dataset_name.startswith("tweeteval"):
        _, task = dataset_name.split("_")
        dataset = TweetEvalDataset(task=task, datasets_dir=TWEETEVAL_DIR)

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return model, dataset


@torch.no_grad()
def compute_image_accuracies(model, dataset):
    device = next(model.parameters()).device
    hits, num_dones = 0, 0
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False)
    pbar = tqdm(dataloader)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=-1)
        hits += (preds == labels).sum().item()
        num_dones += len(images)
        pbar.set_description(f"lambda: {model.lambda_}, accuracy: {hits / num_dones:.3f}")
    return hits / num_dones


@torch.no_grad()
def compute_text_accuracies(model, dataset):
    device = next(model.parameters()).device
    hits, num_dones = 0, 0
    pbar = tqdm(dataset)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        preds = model(**inputs).argmax(dim=-1)
        hits += (preds == labels).sum().item()
        num_dones += 1
        pbar.set_description(f"lambda: {model.lambda_}, accuracy: {hits / num_dones:.3f}")
    return hits / num_dones


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1])
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="_cache")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_file = f"{args.save_dir}/accuracy_vs_smoothing.json"
    if os.path.exists(save_file):
        print(f"File already exists: {save_file}")
        exit()

    # ResNet18
    resnet18_accs = []
    print("Computing ResNet18 accuracies...")
    for lambda_ in args.lambdas:
        resnet18, image_dataset = load_model_dataset("resnet18", "imagenet_2_per_class", lambda_, args.num_samples)
        resnet18.eval().to(args.device)
        resnet18_accs.append(compute_image_accuracies(resnet18, image_dataset))

    # ResNet50
    resnet50_accs = []
    print("Computing ResNet50 accuracies...")
    for lambda_ in args.lambdas:
        resnet50, image_dataset = load_model_dataset("resnet50", "imagenet_2_per_class", lambda_, args.num_samples)
        resnet50.eval().to(args.device)
        resnet50_accs.append(compute_image_accuracies(resnet50, image_dataset))

    # ViT
    vit_accs = []
    print("Computing ViT accuracies...")
    for lambda_ in args.lambdas:
        vit, image_dataset = load_model_dataset("vit", "imagenet_2_per_class", lambda_, args.num_samples)
        vit.eval().to(args.device)
        vit_accs.append(compute_image_accuracies(vit, image_dataset))

    # RoBERTa
    roberta_accs = []
    print("Computing RoBERTa accuracies...")
    for lambda_ in args.lambdas:
        roberta, text_dataset = load_model_dataset("roberta", "tweeteval_sentiment", lambda_, args.num_samples)
        roberta.eval().to(args.device)
        roberta_accs.append(compute_text_accuracies(roberta, text_dataset)) 


    save_dict = {
        "lambdas": args.lambdas,
        "num_smoothing_samples": args.num_samples,
        "image_dataset_size": len(image_dataset),
        "text_dataset_size": len(text_dataset),
        "resnet18": resnet18_accs,  
        "resnet50": resnet50_accs,
        "vit": vit_accs,
        "roberta": roberta_accs,
    }

    with open(save_file, "w") as f:
        json.dump(save_dict, f)

    print(f"Saved to {save_file}")
