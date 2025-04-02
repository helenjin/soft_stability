import os
import sys
import json
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import ViTForImageClassification, RobertaForSequenceClassification
import torchvision

# Load up the project
sys.path.append("../src")
from models import MaskedImageClassifier, MaskedTextClassifier
from data_utils import ImageDataset, TweetEvalDataset

# Load up exlib
EXLIB_PATH = "/home/antonxue/foo/exlib/src"
sys.path.append(EXLIB_PATH)
from exlib.explainers import LimeImageCls, ShapImageCls, IntGradImageCls, MfabaImageCls
from exlib.explainers.lime import LimeTextCls
from exlib.explainers.shap import ShapTextCls
from exlib.explainers.intgrad import IntGradTextCls
from exlib.explainers.mfaba import MfabaTextCls


IMAGENET_SAMPLES_DIR = "/home/antonxue/foo/data/imagenet_samples"
TWEETEVAL_DIR = "/home/antonxue/foo/data/tweeteval/datasets"


def load_model_dataset_explanation(model_name: str, dataset_name: str, explanation_name: str):
    if model_name == "vit":
        model = MaskedImageClassifier(ViTForImageClassification.from_pretrained("google/vit-base-patch16-224"))
    elif model_name == "resnet18":
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
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

    if model_name in ["vit", "resnet18", "resnet50"]:
        if explanation_name == "lime":
            expln_fn = lime_for_image
        elif explanation_name == "shap":
            expln_fn = shap_for_image
        elif explanation_name == "intgrad":
            expln_fn = intgrad_for_image
        elif explanation_name == "mfaba":
            expln_fn = mfaba_for_image
        elif explanation_name == "random":
            expln_fn = random_for_image
        else:
            raise ValueError(f"Explanation {explanation_name} not supported")
    else:
        if explanation_name == "lime":
            expln_fn = lime_for_text
        elif explanation_name == "shap":
            expln_fn = shap_for_text
        elif explanation_name == "intgrad":
            expln_fn = intgrad_for_text
        elif explanation_name == "mfaba":
            expln_fn = mfaba_for_text
        elif explanation_name == "random":
            expln_fn = random_for_text
        else:
            raise ValueError(f"Explanation {explanation_name} not supported")

    return model, dataset, expln_fn


def lime_for_image(model, image, num_samples: int = 100):
    explainer = LimeImageCls(
        model,
        explain_instance_kwargs={ "num_samples": num_samples },
        get_image_and_mask_kwargs={ "positive_only": False },
        LimeImageExplainerKwargs={ "random_state": 1 }
    )

    image = image.view(1, 3, 224, 224)
    pred = model(image).argmax(dim=-1)
    expln = explainer(image, pred)
    attrs = F.avg_pool2d(expln.attributions, kernel_size=16, stride=16).view(14,14)
    return attrs


def shap_for_image(model, image, num_samples: int = 100):
    explainer = ShapImageCls(
        model,
        shap_explainer_kwargs={ "max_evals": num_samples }
    )

    image = image.view(1, 3, 224, 224)
    pred = model(image).argmax(dim=-1)
    expln = explainer(image, pred)
    attrs = F.avg_pool2d(expln.attributions, kernel_size=16, stride=16).mean(dim=1).view(14,14)
    return attrs


def intgrad_for_image(model, image, num_steps: int = 16):
    explainer = IntGradImageCls(model, num_steps=num_steps)
    image = image.view(1, 3, 224, 224)
    pred = model(image).argmax(dim=-1)
    expln = explainer(image, pred)
    attrs = F.avg_pool2d(expln.attributions, kernel_size=16, stride=16).mean(dim=1).view(14,14)
    return attrs


def mfaba_for_image(model, image):
    explainer = MfabaImageCls(model)
    image = image.view(1, 3, 224, 224)
    pred = model(image).argmax(dim=-1)
    expln = explainer(image, pred)
    attrs = F.avg_pool2d(expln.attributions, kernel_size=16, stride=16).mean(dim=1).view(14,14)
    return attrs


def random_for_image(model, image):
    return torch.randn(14, 14, device=image.device)


def lime_for_text(model, input_ids, tokenizer, num_samples: int = 100):
    pred = model(input_ids=input_ids).argmax(dim=-1)
    explainer = LimeTextCls(
        model,
        tokenizer=tokenizer,
        LimeTextExplainerKwargs={
            "mask_string": tokenizer.mask_token,
            "split_expression": lambda x: x.split(),
            "feature_selection": 'none'
        },
        explain_instance_kwargs={
            "num_samples": num_samples,
        },
    )

    expln = explainer(input_ids, pred)
    attrs = expln.attributions
    return attrs.view(-1)


def shap_for_text(model, input_ids, tokenizer, num_samples: int = 100):
    explainer = ShapTextCls(
        model,
        tokenizer=tokenizer,
        pad_value=tokenizer.pad_token_id,
        shap_explainer_kwargs={ "max_evals": num_samples },
        special_tokens=['<s>', '</s>']
    )
    pred = model(input_ids=input_ids).argmax(dim=-1)
    expln = explainer(input_ids, pred)
    attrs = expln.attributions
    return attrs.view(-1)


def intgrad_for_text(model, input_ids, tokenizer, num_steps: int = 16):
    assert hasattr(model, "get_input_embeddings"), "Model must have a get_input_embeddings method"
    explainer = IntGradTextCls(
        model,
        projection_layer=model.get_input_embeddings()
    )
    pred = model(input_ids=input_ids).argmax(dim=-1)
    expln = explainer(input_ids, pred)
    attrs = expln.attributions
    return attrs.view(-1)


def mfaba_for_text(model, input_ids, tokenizer):
    assert hasattr(model, "get_input_embeddings"), "Model must have a get_input_embeddings method"
    explainer = MfabaTextCls(
        model,
        projection_layer=model.get_input_embeddings()
    )
    pred = model(input_ids=input_ids).argmax(dim=-1)
    expln = explainer(input_ids, pred)
    attrs = expln.attributions
    return attrs.view(-1)


def random_for_text(model, input_ids, tokenizer):
    return torch.randn(input_ids.shape[1], device=input_ids.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--explanation_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="_cache/")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, dataset, expln_fn = load_model_dataset_explanation(args.model_name, args.dataset_name, args.explanation_name)
    model.eval().to(args.device)

    # Generate explanations
    all_attrs = []
    pbar = tqdm(dataset)
    for i, item in enumerate(pbar):
        if args.model_name in ["vit", "resnet18", "resnet50"]:
            image = item.to(args.device)
            attrs = expln_fn(model, image)
        else:
            inputs, _ = item
            input_ids = inputs["input_ids"].to(args.device)
            attrs = expln_fn(model, input_ids, dataset.tokenizer)

        all_attrs.append(attrs.cpu().view(-1).tolist())
        pbar.set_description(
            f"{args.model_name} {args.dataset_name} {args.explanation_name} {i+1}/{len(dataset)}"
        )
    
    save_dict = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "explanation_name": args.explanation_name,
        "total_samples": len(dataset),
        "attrs": all_attrs
    }

    save_to_file = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset_name}_{args.explanation_name}_attributions.json")
    with open(save_to_file, "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"Saved explanations to {save_to_file}")
