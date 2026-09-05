# -*- coding: utf-8 -*-
"""
复现 [28] YOLOV8-CMS 的核心任务: 在 CitrusUAT 上做柑橘叶病(12类)分类。
说明: 论文的 CSPPC/MultiDimen/SpatialConv 等自定义模块无开源代码, 无法逐模块复现;
      本脚本用 ultralytics 官方 YOLOv8n-cls 在同一 CitrusUAT 数据上复现"分类任务"本身,
      作为可对照的 baseline (论文报告多类 acc≈97.9%)。
"""
import os
ROOT = r"C:\Users\Windows11\Desktop\新版柑橘\复现_论文"


def main():
    os.chdir(ROOT)
    from ultralytics import YOLO
    model = YOLO("yolov8n-cls.pt")   # 官方预训练分类权重
    model.train(
        data=os.path.join(ROOT, "citrusuat_cls"),
        epochs=40, imgsz=224, batch=32, device=0, workers=0,   # Windows: workers=0 避免多进程
        project="results_yolov8cls", name="citrusuat", exist_ok=True, verbose=True,
    )
    metrics = model.val()
    print("=== FINAL ===")
    print("top1 =", getattr(metrics, "top1", None), "top5 =", getattr(metrics, "top5", None))


if __name__ == "__main__":
    main()
