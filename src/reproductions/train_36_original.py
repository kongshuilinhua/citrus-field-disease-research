# -*- coding: utf-8 -*-
"""
[36] 三通道注意力检测 —— **论文原版复现**(用作者 supp-3 的 ultralyticsmain 原代码)。
模型 = yolov8-powerneck-c2f.yaml(C2f_Attention_iRMB + SimFusion_4in/IFM/power_channel 三件套 Powerneck),
数据 = 本地 SciDB(scidb_det, 15类, 与论文同源数据集)。
依赖: einops、timm(已装)。用 sys.path 优先加载作者改过的 ultralytics, 不污染已装版本。
论文设置: batch64/200epoch/imgsz640/RTX4090; 本机 4060Ti 用 batch16, epoch 可调。
"""
import os, sys
UL = r"C:\Users\Windows11\Desktop\新版柑橘\复现_论文\threechannel_36\ultralytics-main"
# 作者版本根据 YAML 文件名推断规模；使用 yolov8s-* 名称后会加载同目录
# yolov8-powerneck-c2f.yaml，并把 scale 正确设为 s。
YAML = os.path.join(UL, "ultralytics", "cfg", "models", "v8", "yolov8s-powerneck-c2f.yaml")
DATA = r"C:\Users\Windows11\Desktop\新版柑橘\复现_论文\scidb_det\data.yaml"
RUN_ROOT = os.environ.get("CITRUS_RUN_ROOT")
PROJECT = os.path.join(RUN_ROOT, "scidb_det") if RUN_ROOT else r"C:\Users\Windows11\Desktop\新版柑橘\复现_论文\results_scidb_det"
sys.path.insert(0, UL)   # 优先用作者原版 ultralytics


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    from ultralytics import YOLO
    model = YOLO(YAML)   # 原版三通道结构(从头训练)
    if os.environ.get("CITRUS_BUILD_ONLY") == "1":
        params = sum(p.numel() for p in model.model.parameters())
        print("build_only=1 params=", params)
        return
    epochs = int(os.environ.get("CITRUS_EPOCHS", "100"))
    model.train(
        data=DATA, epochs=epochs, imgsz=640, batch=16, device=0, workers=0,
        project=PROJECT,
        name="powerneck_c2f_original", exist_ok=True, patience=30, verbose=True,
    )
    mv = model.val(split="val", project=PROJECT, name="powerneck_c2f_original_val", exist_ok=True)
    print("val  mAP50 =", float(mv.box.map50), "mAP50-95 =", float(mv.box.map))
    mt = model.val(split="test", project=PROJECT, name="powerneck_c2f_original_test", exist_ok=True)
    print("test mAP50 =", float(mt.box.map50), "mAP50-95 =", float(mt.box.map))


if __name__ == "__main__":
    main()
