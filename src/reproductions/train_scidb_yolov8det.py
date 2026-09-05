# -*- coding: utf-8 -*-
"""
复现 [36]《三通道聚合注意力柑橘黄龙病检测》的核心任务：在福建 SciDB 田间集上做检测。
说明：论文的 C2f-IRMB / Powerneck 三通道注意力等自定义模块的代码在 PeerJ 补充材料(s003)，
      若已放到 threechannel_36/ 则可另行接入；本脚本先用官方 YOLOv8n 在**同一 SciDB 数据**
      上复现检测任务，作为可对照 baseline(论文报告 mAP50≈97%)。
数据：extract_scidb.py 生成的 scidb_det/(15 类，train 7192 / val 898 / test 602)。
"""
import os
ROOT = r"C:\Users\Windows11\Desktop\新版柑橘\复现_论文"
RUN_ROOT = os.environ.get("CITRUS_RUN_ROOT")
PROJECT = os.path.join(RUN_ROOT, "scidb_det") if RUN_ROOT else "results_scidb_det"
os.chdir(ROOT)
from ultralytics import YOLO

DATA = os.path.join(ROOT, "scidb_det", "data.yaml")


def main():
    epochs = int(os.environ.get("CITRUS_EPOCHS", "30"))
    model = YOLO("yolov8n.pt")               # 官方检测预训练权重
    model.train(
        data=DATA, epochs=epochs, imgsz=640, batch=16, device=0,
        project=PROJECT, name="yolov8n", exist_ok=True, verbose=True,
        patience=15, workers=0,               # Windows 禁用多进程，避免派生进程卡死
    )
    print("=== val 指标 ===")
    mv = model.val(split="val", project=PROJECT, name="yolov8n_val", exist_ok=True)
    print("val  mAP50 =", float(mv.box.map50), " mAP50-95 =", float(mv.box.map))
    print("=== test 指标 ===")
    mt = model.val(split="test", project=PROJECT, name="yolov8n_test", exist_ok=True)
    print("test mAP50 =", float(mt.box.map50), " mAP50-95 =", float(mt.box.map))


if __name__ == "__main__":
    main()
