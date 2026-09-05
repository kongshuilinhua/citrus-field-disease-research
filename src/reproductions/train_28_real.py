# -*- coding: utf-8 -*-
"""
[28] YOLOV8-CMS 分类 —— "论文真方法(重建)"版训练。
用重建的 CSPPC + SpatialConv + MultiDimen 组成的 yolov8n-cms.yaml，在 CitrusUAT 上训分类。
⚠ 论文模块细节未公开(最低保真度重建)，自定义结构从头训练；
  与 train_citrusuat_yolov8cls.py(官方YOLOv8n-cls baseline) 形成对照。
"""
import os, sys
ROOT = r"C:\Users\Windows11\Desktop\新版柑橘\复现_论文"
RUN_ROOT = os.environ.get("CITRUS_RUN_ROOT")
PROJECT = os.path.join(RUN_ROOT, "yolov8cls") if RUN_ROOT else "results_yolov8cls"
os.chdir(ROOT); sys.path.insert(0, ROOT)
from custom_modules.register import register
register()
from ultralytics import YOLO


def main():
    epochs = int(os.environ.get("CITRUS_EPOCHS", "60"))
    model = YOLO(os.path.join(ROOT, "yolov8n-cms.yaml"), task="classify")   # 从头(自定义结构)
    model.train(data=os.path.join(ROOT, "citrusuat_cls"),
                epochs=epochs, imgsz=224, batch=32, device=0,
                project=PROJECT, name="citrusuat_cms", exist_ok=True,
                patience=20, workers=0, verbose=True)
    m = model.val(project=PROJECT, name="citrusuat_cms_val", exist_ok=True)
    print("top1 =", getattr(m, "top1", None), "top5 =", getattr(m, "top5", None))


if __name__ == "__main__":
    main()
