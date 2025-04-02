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
from models import CertifiedMuSImageClassifier, CertifiedMuSTextClassifier
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
        model = CertifiedMuSImageClassifier(
            ViTForImageClassification.from_pretrained("google/vit-base-patch16-224"),
            lambda_=0.25,
            quant=64
        )
    elif model_name == "resnet18":
        model = CertifiedMuSImageClassifier(
            tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1),
            lambda_=0.25,
            quant=64
        )
    elif model_name == "resnet50":
        model = CertifiedMuSImageClassifier(
            tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1),
            lambda_=0.25,
            quant=64
        )
    elif model_name == "roberta":
        _, task = dataset_name.split("_")
        model = CertifiedMuSTextClassifier(
            RobertaForSequenceClassification.from_pretrained(f"cardiffnlp/roberta-base-{task}"),
            lambda_=0.25,
            quant=64
        )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--explanation_name", type=str, required=True)
    parser.add_argument("--top_k_frac", type=float, default=0.25)
    parser.add_argument("--save_dir", type=str, default="_cache/")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, dataset, attrs = load_model_dataset_attributions(args.save_dir, args.model_name, args.dataset_name, args.explanation_name, args.top_k_frac)
    model.eval().to(args.device)

    all_certified_radii = []
    pbar = tqdm(dataset)
    for i, item in enumerate(pbar):
        if args.model_name in ["vit", "resnet18", "resnet50"]:
            image = item.to(args.device)
            attr = attrs[i].to(args.device)
            out = model(image.unsqueeze(0), attr.unsqueeze(0))
            all_certified_radii.append(out["cert_rs"].cpu().item())
        else:
            inputs, _ = item
            input_ids = inputs["input_ids"].to(args.device)
            attr = attrs[i].to(args.device)
            out = model(input_ids.unsqueeze(0), attr.unsqueeze(0))
            all_certified_radii.append(out["cert_rs"].cpu().item())

        pbar.set_description(
            f"{args.model_name} {args.dataset_name} {args.explanation_name} {i+1}/{len(dataset)}"
        )

    save_dict = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "explanation_name": args.explanation_name,
        "total_samples": len(dataset),
        "lambda": model.lambda_,
        "quant": model.q,
        "certified_radii": all_certified_radii
    }

    save_file = os.path.join(args.save_dir, f"{args.model_name}_{args.dataset_name}_{args.explanation_name}_hard_stability_radii.json")
    with open(save_file, "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"Saved certified radii to {save_file}")
