# -*- coding: utf-8 -*-
"""构建自检：注册自定义模块 -> 用重建版 YAML 建模型 -> 假输入前向，确认可用。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from custom_modules.register import register
register()
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
cases = [
    ("yolov8n-3ch.yaml", "detect", 640, "[36] 三通道检测"),
    ("yolov8n-cms.yaml", "classify", 224, "[28] CMS 分类"),
]
for yml, task, imgsz, tag in cases:
    print(f"\n===== 构建 {tag}: {yml} ({task}) =====")
    m = YOLO(os.path.join(ROOT, yml), task=task)
    n_par = sum(p.numel() for p in m.model.parameters())
    print(f"  参数量 = {n_par/1e6:.2f}M")
    m.model.eval()
    with torch.no_grad():
        y = m.model(torch.randn(1, 3, imgsz, imgsz))
    def shapes(o):
        if hasattr(o, "shape"): return tuple(o.shape)
        if isinstance(o, (list, tuple)): return [shapes(e) for e in o]
        return type(o).__name__
    print(f"  前向 OK, 输出形状={shapes(y)}")
print("\n全部自定义模型构建 + 前向通过 ✅")
