import os
import sys
import json
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.models as tvm
from torch.utils.data import Subset
from transformers import ViTForImageClassification, RobertaForSequenceClassification

# Load up the project
sys.path.append("../src")

from models import SmoothMaskedImageClassifier, SmoothMaskedTextClassifier
from data_utils import ImageDataset, TweetEvalDataset
from stability import soft_stability_rate, soft_stability_rate_text

IMAGENET_SAMPLES_DIR = "/home/antonxue/foo/data/imagenet_samples"
TWEETEVAL_DIR = "/home/antonxue/foo/data/tweeteval/datasets"


def load_model(model_name: str, lambda_: float, num_samples: int = 16, dataset_name: str = None):
    if model_name == "vit":
        model = SmoothMaskedImageClassifier(
            ViTForImageClassification.from_pretrained("google/vit-base-patch16-224"),
            lambda_=lambda_,
            num_samples=num_samples
        )
    elif model_name == "resnet18":
        model = SmoothMaskedImageClassifier(
            tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1),
            lambda_=lambda_,
            num_samples=num_samples
        )   
    elif model_name == "resnet50":
        model = SmoothMaskedImageClassifier(
            tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1),
            lambda_=lambda_,
            num_samples=num_samples
        )
    elif model_name == "roberta":
        _, task = dataset_name.split("_")
        model= SmoothMaskedTextClassifier(
            RobertaForSequenceClassification.from_pretrained(f"cardiffnlp/roberta-base-{task}"),
            lambda_=lambda_,
            num_samples=num_samples
        )
    else:
        raise ValueError(f"Model {model_name} not supported")

    return model


def load_dataset(dataset_name: str):
    if dataset_name.startswith("imagenet") and dataset_name.endswith("per_class"):
        dataset =ImageDataset(
            IMAGENET_SAMPLES_DIR + "/" + dataset_name,
            image_size=(224, 224),
            use_preprocessor=True
        )   
    elif dataset_name.startswith("tweeteval"):
        _, task = dataset_name.split("_")
        dataset = TweetEvalDataset(
            task=task,
            datasets_dir=TWEETEVAL_DIR,
        )  
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return dataset


def compute_stability_vs_smoothing(model, image, attr, radii: list[int]):
    soft_stability_rates = []
    for radius in radii:
        rate = soft_stability_rate(model, image, attr, radius, epsilon=0.1, delta=0.1)
        soft_stability_rates.append(round(rate.item(), 4))
    return soft_stability_rates


def compute_stability_vs_smoothing_text(model, input_ids, attr, radii: list[int]):
    attention_mask = torch.ones_like(input_ids)
    soft_stability_rates = []
    for radius in radii:
        rate = soft_stability_rate_text(model, input_ids, attention_mask, attr, radius, epsilon=0.1, delta=0.1)
        soft_stability_rates.append(round(rate.item(), 4))
    return soft_stability_rates


def lambda_to_key(lambda_: float):
    return f"{lambda_:.3f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument("--max_samples_from_dataset", type=int, default=10)
    parser.add_argument("--lambdas", nargs='+', default=[1.0, 0.8, 0.6, 0.4, 0.2])
    parser.add_argument("--topk_frac", type=float, default=0.25)
    parser.add_argument("--save_dir", type=str, default="_cache/")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    save_file = os.path.join(args.save_dir, f"{args.model_name}_{args.dataset_name}_stability_vs_smoothing.json")
    if os.path.exists(save_file):
        print(f"File already exists: {save_file}")
        exit()

    # Take a random subset of the dataset, of size args.max_samples_from_dataset
    torch.manual_seed(1234)
    dataset = load_dataset(args.dataset_name)
    if args.model_name in ["vit", "resnet18", "resnet50"]:
        chosen_indices = torch.randperm(len(dataset))[:args.max_samples_from_dataset]
        dataset = Subset(dataset, chosen_indices)
    else:
        # Find all the "good" indices, which are the indices of the samples with at least 50 tokens
        good_indices = [i for i in range(len(dataset)) if dataset[i][0]["input_ids"].numel() >= 50]
        good_indices = torch.tensor(good_indices)
        chosen_indices = good_indices[torch.randperm(len(good_indices))[:args.max_samples_from_dataset]]
        dataset = Subset(dataset, chosen_indices)

    # Define the radii of perturbation
    if args.model_name in ["vit", "resnet18", "resnet50"]:
        all_radii = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        all_radii = [r for r in all_radii if r <= 196 * (1 - args.topk_frac)]
    else:
        all_radii = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    lambda_to_rates = {lambda_to_key(lambda_): [] for lambda_ in args.lambdas}
    for lambda_ in args.lambdas:
        model = load_model(args.model_name, lambda_, dataset_name=args.dataset_name)
        model.to(args.device)
        model.eval()

        pbar = tqdm(dataset)
        for item in pbar:
            if args.model_name in ["vit", "resnet18", "resnet50"]:
                image = item.to(args.device)
                attr = torch.randn(196, device=args.device)
                attr = (attr > attr.quantile(1 - args.topk_frac)).long()
                rates = compute_stability_vs_smoothing(model, image, attr, all_radii)
            else:
                inputs, _ = item
                input_ids = inputs["input_ids"].to(args.device)
                attr = torch.randn(*input_ids.shape, device=args.device)
                attr = (attr > attr.quantile(1 - args.topk_frac)).long()
                rates = compute_stability_vs_smoothing_text(model, input_ids, attr, all_radii)

            lambda_to_rates[lambda_to_key(lambda_)].append(rates)
            pbar.set_description(f"{args.model_name} {args.dataset_name} {lambda_:.3f}")

    save_dict = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "topk_frac": args.topk_frac,
        "radii": all_radii,
        "lambdas": args.lambdas,
        "max_samples_from_dataset": args.max_samples_from_dataset,
        "lambda_to_rates": lambda_to_rates
    }

    with open(save_file, "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"Saved stability vs smoothing to {save_file}")
