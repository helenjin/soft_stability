import os
import sys
import json
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.models as tvm
from transformers import ViTForImageClassification, RobertaForSequenceClassification

# Load up the project
sys.path.append("../src")
from models import MaskedImageClassifier, MaskedTextClassifier
from data_utils import ImageDataset, TweetEvalDataset
from stability import soft_stability_rate, soft_stability_rate_text


IMAGENET_SAMPLES_DIR = "/home/antonxue/foo/data/imagenet_samples"
TWEETEVAL_DIR = "/home/antonxue/foo/data/tweeteval/datasets"

IMAGE_RADII = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 147]


def load_model_dataset_attributions(
    save_dir: str,
    model_name: str,
    dataset_name: str,
    explanation_name: str,
    top_k_frac: float = 0.25
):
    if model_name == "vit":
        model = MaskedImageClassifier(ViTForImageClassification.from_pretrained("google/vit-base-patch16-224"))
    elif model_name == "resnet18":
        model = MaskedImageClassifier(tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1))
    elif model_name == "resnet50":
        model = MaskedImageClassifier(tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1))
    elif model_name == "roberta":
        _, task = dataset_name.split("_")
        model = MaskedTextClassifier(RobertaForSequenceClassification.from_pretrained(f"cardiffnlp/roberta-base-{task}"))
    else:
        raise ValueError(f"Model {model_name} not supported")

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

    save_file = os.path.join(save_dir, f"{model_name}_{dataset_name}_{explanation_name}_attributions.json")
    save_dict = json.load(open(save_file))

    attrs = []
    for attr in save_dict["attrs"]:
        attr = torch.tensor(attr).view(-1)
        attrs.append((attr > attr.quantile(1 - top_k_frac)).long())

    return model, dataset, attrs


def compute_soft_stability_image(model, image, attr):
    soft_stability_rates = []
    for radius in IMAGE_RADII:
        rate = soft_stability_rate(model, image, attr, radius, epsilon=0.1, delta=0.1)
        soft_stability_rates.append(round(rate.item(), 4))
    return soft_stability_rates


def compute_soft_stability_text(model, input_ids, attr):
    attention_mask = torch.ones_like(input_ids)
    soft_stability_rates = []
    max_radius_plus1 = min(21, attr.numel() - attr.sum())
    for radius in range(1, max_radius_plus1):
        rate = soft_stability_rate_text(model, input_ids, attention_mask, attr, radius, epsilon=0.1, delta=0.1)
        soft_stability_rates.append(round(rate.item(), 4))
    return soft_stability_rates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--explanation_name", type=str, required=True)
    parser.add_argument("--top_k_frac", type=float, default=0.25)
    parser.add_argument("--save_dir", type=str, default="_cache/")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    save_file = os.path.join(args.save_dir, f"{args.model_name}_{args.dataset_name}_{args.explanation_name}_soft_stability_rates.json")
    if os.path.exists(save_file):
        print(f"File already exists: {save_file}")
        exit()

    model, dataset, attrs = load_model_dataset_attributions(args.save_dir, args.model_name, args.dataset_name, args.explanation_name, args.top_k_frac)
    model.eval().to(args.device)

    all_soft_stability_rates = []
    pbar = tqdm(dataset)
    for i, item in enumerate(pbar):
        if args.model_name in ["vit", "resnet18", "resnet50"]:
            image = item.to(args.device)
            attr = attrs[i].to(args.device)
            soft_stability_rates = compute_soft_stability_image(model, image, attr)
        else:
            inputs, _ = item
            input_ids = inputs["input_ids"].to(args.device)
            attr = attrs[i].to(args.device)
            soft_stability_rates = compute_soft_stability_text(model, input_ids, attr)

        all_soft_stability_rates.append(soft_stability_rates)
        pbar.set_description(f"{args.model_name} {args.dataset_name} {args.explanation_name}")

    save_dict = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "explanation_name": args.explanation_name,
        "soft_stability_rates": all_soft_stability_rates
    }

    with open(save_file, "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"Saved soft stability rates to {save_file}")
