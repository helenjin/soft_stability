import os
import sys
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import Subset
import torchvision.models as tvm
from transformers import ViTForImageClassification, RobertaForSequenceClassification

# Load up the project
sys.path.append("../src")

from models import SmoothedImageClassifier, SmoothedTextClassifier, BinarizedMaskedImageClassifier, BinarizedMaskedTextClassifier
from banal import sample_level_k_fourier_info
from data_utils import ImageDataset, TweetEvalDataset


IMAGENET_SAMPLES_DIR = "/home/antonxue/foo/data/imagenet_samples"
TWEETEVAL_DIR = "/home/antonxue/foo/data/tweeteval/datasets"


def load_model(model_name: str, dataset_name: str, lambda_: float):
    if model_name == "vit":
        raw_model = SmoothedImageClassifier(
            ViTForImageClassification.from_pretrained("google/vit-base-patch16-224"),
            lambda_=lambda_,
            num_samples = 16,
            grid_size=(7,7)
        )
    elif model_name == "resnet18":
        raw_model = SmoothedImageClassifier(
            tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1),
            lambda_=lambda_,
            num_samples=16,
            grid_size=(7,7)
        )
    elif model_name == "resnet50":
        raw_model = SmoothedImageClassifier(
            tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1),
            lambda_=lambda_,
            num_samples=16,
            grid_size=(7,7)
        )
    elif model_name == "roberta":
        _, task = dataset_name.split("_")
        raw_model = SmoothedTextClassifier(
            RobertaForSequenceClassification.from_pretrained(f"cardiffnlp/roberta-base-{task}"),
            lambda_=lambda_,
            num_samples=16
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return raw_model
            

def load_dataset(dataset_name: str):
    if dataset_name.startswith("imagenet") and dataset_name.endswith("per_class"):
        dataset = ImageDataset(
            IMAGENET_SAMPLES_DIR + "/" + dataset_name,
            image_size=(224, 224),
            use_preprocessor=True
        )

    elif dataset_name.startswith("tweeteval"):
        _, task = dataset_name.split("_")
        dataset = TweetEvalDataset(
            task=task,
            datasets_dir=TWEETEVAL_DIR
        )

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return dataset


def calculate_spectrum(f, n):
    """
    Calculate the spectrum info of a model for an image.
    """
    all_infos = []
    pbar = tqdm(range(0, n+1))
    for k in pbar:
        all_infos.append(sample_level_k_fourier_info(f, n, k))
        pbar.set_description(f"Calculating spectrum info for k={k}")
    return all_infos


def lambda_to_key(lambda_: float):
    return f"{lambda_:.3f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--lambdas", nargs='+', default=[1.0, 0.8, 0.6, 0.4, 0.2])
    parser.add_argument("--save_dir", type=str, default="_cache/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples_from_dataset", type=int, default=10)
    args = parser.parse_args()

    save_file = os.path.join(args.save_dir, f"{args.model_name}_{args.dataset_name}_fourier_vs_smoothing.json")
    if os.path.exists(save_file):
        print(f"File already exists: {save_file}")
        exit()

    torch.manual_seed(1234)
    dataset = load_dataset(args.dataset_name)
    if args.model_name in ["vit", "resnet18", "resnet50"]:
        chosen_indices = torch.randperm(len(dataset))[:args.max_samples_from_dataset]
        dataset = Subset(dataset, chosen_indices)
    elif args.model_name == "roberta":
        good_indices = [i for i in range(len(dataset)) if 30 <= dataset[i][0]["input_ids"].numel() <= 40]
        good_indices = torch.tensor(good_indices)
        chosen_indices = good_indices[torch.randperm(len(good_indices))[:args.max_samples_from_dataset]]
        dataset = Subset(dataset, chosen_indices)
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    lambda_to_avg_mass = {lambda_to_key(lambda_): [] for lambda_ in args.lambdas}
    lambda_to_max_mass = {lambda_to_key(lambda_): [] for lambda_ in args.lambdas}
    lambda_to_avg_variance = {lambda_to_key(lambda_): [] for lambda_ in args.lambdas}
    lambda_to_max_variance = {lambda_to_key(lambda_): [] for lambda_ in args.lambdas}

    pbar = tqdm(dataset)
    for lambda_ in args.lambdas:
        print(f"Calculating spectrum info for lambda={lambda_}")    
        raw_model = load_model(args.model_name, args.dataset_name, lambda_)

        for i, item in enumerate(pbar):
            if args.model_name in ["vit", "resnet18", "resnet50"]:
                image = item.to(args.device)
                n = raw_model.mask_dim
                binarized_model = BinarizedMaskedImageClassifier(raw_model, image)
                binarized_model = binarized_model.eval().to(args.device)
            else:
                inputs, _ = item
                input_ids = inputs["input_ids"].to(args.device)
                n = input_ids.numel()
                binarized_model = BinarizedMaskedTextClassifier(raw_model, input_ids)
                binarized_model = binarized_model.eval().to(args.device)

            fourier_infos = calculate_spectrum(binarized_model, n)
            avg_mass = [info["average_mass"].max().item() for info in fourier_infos]
            max_mass = [info["average_mass"].mean().item() for info in fourier_infos]
            avg_variance = [info["average_variance"].max().item() for info in fourier_infos]
            max_variance = [info["average_variance"].mean().item() for info in fourier_infos]

            lambda_to_avg_mass[lambda_to_key(lambda_)].append(avg_mass)
            lambda_to_max_mass[lambda_to_key(lambda_)].append(max_mass)
            lambda_to_avg_variance[lambda_to_key(lambda_)].append(avg_variance)
            lambda_to_max_variance[lambda_to_key(lambda_)].append(max_variance)

            pbar.set_description(f"Calculating spectrum info for lambda={lambda_}, on item {i+1}/{len(dataset)}")

    save_dict = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "max_samples_from_dataset": args.max_samples_from_dataset,
        "lambdas": args.lambdas,
        "lambda_to_avg_mass": lambda_to_avg_mass,
        "lambda_to_max_mass": lambda_to_max_mass,
        "lambda_to_avg_variance": lambda_to_avg_variance,
        "lambda_to_max_variance": lambda_to_max_variance,
    }

    with open(save_file, "w") as f:
        json.dump(save_dict, f, indent=2)