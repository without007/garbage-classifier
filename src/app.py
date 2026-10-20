# -*- coding: utf-8 -*-
"""Gradio 演示界面:上传/拍照 → Top-3 预测类别与置信度柱状图。

用法: python -m src.app --model cnn
"""
import argparse

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from .config import CKPT_DIR, CLASS_NAMES_CN, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from .models import build_model

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(name: str, tag: str = ""):
    suffix = f"_{tag}" if tag else ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT_DIR / f"best_{name}{suffix}.pth", map_location=device)
    model = build_model(name).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # 输出索引对应训练时的 ImageFolder 类别顺序(字母序),不能假设等于 config.CLASSES
    return model, device, ckpt["classes"]


def predict(img: Image.Image, model_name: str):
    model, device, classes = MODELS[model_name]
    x = TRANSFORM(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0).cpu()
    top = probs.topk(3)
    labels = [f"{CLASS_NAMES_CN.get(classes[i], classes[i])} ({classes[i]})" for i in top.indices.tolist()]
    return {l: float(p) for l, p in zip(labels, top.values.tolist())}


MODELS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
    ap.add_argument("--tag", default="unfrozen", help="权重标签,默认加载精度最好的解冻版")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    MODELS[args.model] = load_model(args.model, args.tag)

    demo = gr.Interface(
        fn=lambda img: predict(img, args.model),
        inputs=gr.Image(type="pil", label="上传垃圾图片或拍照", sources=["upload", "webcam"]),
        outputs=gr.Label(num_top_classes=3, label="Top-3 预测与置信度"),
        title="生活垃圾智能分类系统",
        description=f"基于 {args.model} 的五类生活垃圾分类演示(塑料 / 金属 / 玻璃 / 纸张 / 厨余)",
    )
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
