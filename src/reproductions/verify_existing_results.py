# -*- coding: utf-8 -*-
"""
用现有权重重新推理，核验历史实验结果是否可由磁盘产物复算。

本脚本不训练模型，也不改写原结果目录。核验记录统一写入：
    复现_论文/实验记录/核验_20260725/

可核验范围：
1. Mendeley 三骨干分类基线；
2. 三骨干在 CitrusUAT 重叠类上的 M0；
3. ViT 四类测试结果；
4. YOLOv8n-cls 在 CitrusUAT val 上的结果；
5. 关键脚本、日志和权重的 SHA-256。

DANN 历史运行没有保存权重，因此只能核对日志，不能重新推理。
"""

import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ROOT = Path(r"C:\Users\Windows11\Desktop\新版柑橘")
BASELINE_RESULTS = ROOT / "复现_baseline" / "results"
MENDELEY_ROOT = ROOT / "数据集_整理" / "源域_Mendeley叶5类"
M0_ROOT = ROOT / "复现_baseline" / "field_test_CitrusUAT"
VIT_ROOT = ROOT / "数据集_整理" / "源域_4类增广"
REPRO_ROOT = ROOT / "复现_论文"
VERIFY_DIR = REPRO_ROOT / "实验记录" / "核验_20260725"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MENDELEY_CLASSES = ["black_spot", "canker", "greening", "healthy", "melanose"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_state(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_baseline(name: str, num_classes: int = 5):
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"未知模型：{name}")
    return model


@torch.inference_mode()
def evaluate_loader(model, loader, device):
    model.eval()
    predictions = []
    labels = []
    for images, target in loader:
        output = model(images.to(device))
        predictions.extend(output.argmax(1).cpu().tolist())
        labels.extend(target.tolist())
    return np.asarray(labels), np.asarray(predictions)


def verify_baselines(device):
    tf = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    dataset = datasets.ImageFolder(MENDELEY_ROOT / "val", tf)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    results = {}

    for name in ["efficientnet_b0", "resnet50", "mobilenet_v3_large"]:
        started = time.perf_counter()
        weight = BASELINE_RESULTS / f"best_{name}.pt"
        model = build_baseline(name).to(device)
        model.load_state_dict(load_state(weight))
        labels, predictions = evaluate_loader(model, loader, device)
        actual = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "sample_count": int(len(labels)),
            "seconds": round(time.perf_counter() - started, 3),
        }
        with (BASELINE_RESULTS / f"summary_{name}.json").open("r", encoding="utf-8") as f:
            claimed = json.load(f)
        actual["历史记录"] = {
            "accuracy": claimed["accuracy"],
            "macro_f1": claimed["macro_f1"],
        }
        actual["一致"] = (
            abs(actual["accuracy"] - claimed["accuracy"]) < 1e-9
            and abs(actual["macro_f1"] - claimed["macro_f1"]) < 1e-9
        )
        results[name] = actual
        print(
            f"[基线] {name}: accuracy={actual['accuracy']:.6f}, "
            f"macro_f1={actual['macro_f1']:.6f}, 一致={actual['一致']}",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def m0_label(folder_name: str):
    lowered = folder_name.lower()
    if "healthy" in lowered:
        return "healthy"
    if "hlb" in lowered or "huanglong" in lowered:
        return "greening"
    return None


class M0Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
        self.samples = []
        for folder in sorted(M0_ROOT.iterdir()):
            if not folder.is_dir():
                continue
            label = m0_label(folder.name)
            if label is None:
                continue
            for path in sorted(folder.rglob("*")):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((path, MENDELEY_CLASSES.index(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.tf(Image.open(path).convert("RGB")), label


@torch.inference_mode()
def verify_m0(device):
    dataset = M0Dataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    with (BASELINE_RESULTS / "m0_crossdomain_summary.json").open("r", encoding="utf-8") as f:
        claimed_all = json.load(f)
    results = {}

    for name in ["efficientnet_b0", "resnet50", "mobilenet_v3_large"]:
        started = time.perf_counter()
        model = build_baseline(name).to(device)
        model.load_state_dict(load_state(BASELINE_RESULTS / f"best_{name}.pt"))
        model.eval()
        labels = []
        predictions = []
        restricted_predictions = []
        for images, target in loader:
            output = model(images.to(device))
            predictions.extend(output.argmax(1).cpu().tolist())
            restricted = output[:, [MENDELEY_CLASSES.index("greening"), MENDELEY_CLASSES.index("healthy")]]
            restricted_predictions.extend(
                [
                    MENDELEY_CLASSES.index("greening") if value == 0 else MENDELEY_CLASSES.index("healthy")
                    for value in restricted.argmax(1).cpu().tolist()
                ]
            )
            labels.extend(target.tolist())
        strict5 = float(accuracy_score(labels, predictions))
        restricted2 = float(accuracy_score(labels, restricted_predictions))
        claimed = claimed_all[name]
        item = {
            "strict5_acc": strict5,
            "restricted2_acc": restricted2,
            "sample_count": len(labels),
            "seconds": round(time.perf_counter() - started, 3),
            "历史记录": {
                "strict5_acc": claimed["strict5_acc"],
                "restricted2_acc": claimed["restricted2_acc"],
            },
            "一致": (
                abs(strict5 - claimed["strict5_acc"]) < 1e-9
                and abs(restricted2 - claimed["restricted2_acc"]) < 1e-9
            ),
        }
        results[name] = item
        print(
            f"[M0] {name}: strict5={strict5:.6f}, restricted2={restricted2:.6f}, "
            f"一致={item['一致']}",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def verify_vit(device):
    started = time.perf_counter()
    tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    dataset = datasets.ImageFolder(VIT_ROOT / "test", tf)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, len(dataset.classes))
    model.load_state_dict(load_state(REPRO_ROOT / "results_vit" / "best_vit_b16.pt"))
    model = model.to(device)
    labels, predictions = evaluate_loader(model, loader, device)
    accuracy = float(accuracy_score(labels, predictions))
    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    item = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "sample_count": len(labels),
        "seconds": round(time.perf_counter() - started, 3),
        "历史记录": {"accuracy": 0.9798, "macro_f1": 0.9773},
        "一致_按日志四位小数": round(accuracy, 4) == 0.9798 and round(macro_f1, 4) == 0.9773,
    }
    print(
        f"[ViT] accuracy={accuracy:.6f}, macro_f1={macro_f1:.6f}, "
        f"一致={item['一致_按日志四位小数']}",
        flush=True,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return item


def verify_yolo_cls():
    started = time.perf_counter()
    from ultralytics import YOLO

    weight = REPRO_ROOT / "results_yolov8cls" / "citrusuat" / "weights" / "best.pt"
    model = YOLO(str(weight))
    metrics = model.val(
        data=str(REPRO_ROOT / "citrusuat_cls"),
        split="val",
        imgsz=224,
        batch=32,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0,
        project=str(VERIFY_DIR / "yolo_validation"),
        name="citrusuat",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    top1 = float(metrics.top1)
    top5 = float(metrics.top5)
    with (REPRO_ROOT / "results_yolov8cls" / "citrusuat" / "results.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    historical_top1 = float(last["metrics/accuracy_top1"])
    historical_top5 = float(last["metrics/accuracy_top5"])
    item = {
        "top1": top1,
        "top5": top5,
        "sample_count": 190,
        "seconds": round(time.perf_counter() - started, 3),
        "历史记录": {"top1": historical_top1, "top5": historical_top5},
        # results.csv 只保留有限位小数，按其记录精度比较。
        "一致": abs(top1 - historical_top1) < 1e-5 and abs(top5 - historical_top5) < 1e-5,
    }
    print(f"[YOLOv8n-cls] top1={top1:.6f}, top5={top5:.6f}, 一致={item['一致']}", flush=True)
    return item


def artifact_hashes():
    paths = [
        ROOT / "复现_baseline" / "train_cls_baseline.py",
        ROOT / "复现_baseline" / "eval_m0_crossdomain.py",
        BASELINE_RESULTS / "best_efficientnet_b0.pt",
        BASELINE_RESULTS / "best_resnet50.pt",
        BASELINE_RESULTS / "best_mobilenet_v3_large.pt",
        BASELINE_RESULTS / "m0_crossdomain_summary.json",
        REPRO_ROOT / "dann_uda_reproduce.py",
        REPRO_ROOT / "results_dann" / "dann_log.txt",
        REPRO_ROOT / "train_vit_citrus.py",
        REPRO_ROOT / "results_vit" / "best_vit_b16.pt",
        REPRO_ROOT / "results_vit" / "vit_log.txt",
        REPRO_ROOT / "train_citrusuat_yolov8cls.py",
        REPRO_ROOT / "results_yolov8cls" / "citrusuat" / "results.csv",
        REPRO_ROOT / "results_yolov8cls" / "citrusuat" / "weights" / "best.pt",
    ]
    return {
        str(path.relative_to(ROOT)): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "modified": path.stat().st_mtime,
        }
        for path in paths
        if path.exists()
    }


def main():
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    print(f"核验设备：{device}", flush=True)

    report = {
        "核验时间": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "说明": "用现有权重重新推理；不重新训练，不改写原结果。",
        "环境": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "基线": verify_baselines(device),
        "M0": verify_m0(device),
        "ViT": verify_vit(device),
        "YOLOv8n_cls": verify_yolo_cls(),
        "DANN": {
            "状态": "不可做权重级复核",
            "原因": "历史脚本只保存日志，没有保存模型权重。",
        },
        "产物哈希": artifact_hashes(),
    }
    report["总耗时秒"] = round(time.perf_counter() - started, 3)

    output = VERIFY_DIR / "verification.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"核验记录：{output}", flush=True)


if __name__ == "__main__":
    main()
