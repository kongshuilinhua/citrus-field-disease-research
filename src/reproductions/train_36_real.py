# -*- coding: utf-8 -*-
"""
[36] 三通道注意力检测 —— "论文真方法(重建)"版训练。
用重建的 C2f_iRMB + ThreeChAgg 组成的 yolov8n-3ch.yaml，在 SciDB 上训检测。
⚠ 自定义结构无预训练权重 -> 从头训练，需较多轮数；且模块按论文文字重建，
  不保证还原论文 mAP。与 train_scidb_yolov8det.py(标准YOLOv8 baseline) 形成对照。
"""
import os, sys
ROOT = r"C:\Users\Windows11\Desktop\新版柑橘\复现_论文"
RUN_ROOT = os.environ.get("CITRUS_RUN_ROOT")
PROJECT = os.path.join(RUN_ROOT, "scidb_det") if RUN_ROOT else "results_scidb_det"
os.chdir(ROOT); sys.path.insert(0, ROOT)
from custom_modules.register import register
register()
from ultralytics import YOLO


def main():
    epochs = int(os.environ.get("CITRUS_EPOCHS", "30"))
    model = YOLO(os.path.join(ROOT, "yolov8n-3ch.yaml"), task="detect")   # 从头(自定义结构)
    model.train(data=os.path.join(ROOT, "scidb_det", "data.yaml"),
                epochs=epochs, imgsz=640, batch=16, device=0,
                project=PROJECT, name="yolov8n_3ch", exist_ok=True,
                patience=20, workers=0, verbose=True)
    mv = model.val(split="val", project=PROJECT, name="yolov8n_3ch_val", exist_ok=True)
    print("val  mAP50 =", float(mv.box.map50), "mAP50-95 =", float(mv.box.map))
    mt = model.val(split="test", project=PROJECT, name="yolov8n_3ch_test", exist_ok=True)
    print("test mAP50 =", float(mt.box.map50), "mAP50-95 =", float(mt.box.map))


if __name__ == "__main__":
    main()
