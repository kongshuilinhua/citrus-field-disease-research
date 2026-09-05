# -*- coding: utf-8 -*-
"""
[36]《三通道聚合注意力柑橘黄龙病检测》自定义模块 —— 按论文文字描述**重建**。
⚠ 非官方代码(官方代码在 PeerJ 补充材料 s003)。这里根据论文对
  "C2f Attention inverse residual moving block (C2f-IRMB)" 与
  "三通道聚合注意力 Powerneck" 的描述做合理重建, 不保证还原其精确结构/指标。
"""
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, C2f


class iRMB(nn.Module):
    """倒残差移动块(重建 iRMB/EMO 思路), 通道保持, 供 C2f 内部替换 Bottleneck。"""
    def __init__(self, c1, c2, shortcut=True, e=2.0):
        super().__init__()
        ch = int(c2 * e)
        self.add = shortcut and c1 == c2
        self.pw1 = Conv(c1, ch, 1, 1)              # 1x1 扩张
        self.dw = Conv(ch, ch, 3, 1, g=ch)         # 深度卷积
        r = max(8, ch // 4)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, r, 1), nn.SiLU(),
                                nn.Conv2d(r, ch, 1), nn.Sigmoid())   # 轻量注意力
        self.pw2 = Conv(ch, c2, 1, 1, act=False)   # 1x1 投影
    def forward(self, x):
        y = self.pw1(x)
        y = self.dw(y)
        y = y * self.se(y)
        y = self.pw2(y)
        return x + y if self.add else y


class C2f_iRMB(C2f):
    """C2f 但把 Bottleneck 换成 iRMB(重建 [36] 的 C2f-IRMB)。签名与 C2f 一致。"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(iRMB(self.c, self.c, shortcut) for _ in range(n))


class ThreeChAgg(nn.Module):
    """三通道聚合注意力(重建 [36] Powerneck): 通道 / 空间 / 坐标 三支, 学习权重聚合, 通道保持。"""
    def __init__(self, c1, c2):
        super().__init__()
        assert c1 == c2, "ThreeChAgg 通道保持, 需 c1==c2"
        c = c1
        r = max(8, c // 8)
        # 支1: 通道注意力
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, r, 1), nn.SiLU(), nn.Conv2d(r, c, 1))
        # 支2: 空间注意力
        self.sa = nn.Conv2d(2, 1, 7, padding=3)
        # 支3: 坐标注意力(H/W 方向池化)
        self.ph = nn.AdaptiveAvgPool2d((None, 1))
        self.pw = nn.AdaptiveAvgPool2d((1, None))
        self.coord = nn.Conv2d(c, c, 1)
        self.w = nn.Parameter(torch.ones(3))
    def forward(self, x):
        a1 = torch.sigmoid(self.ca(x))                                  # (B,C,1,1)
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        a2 = torch.sigmoid(self.sa(s))                                  # (B,1,H,W)
        a3 = torch.sigmoid(self.coord(self.ph(x) * self.pw(x)))         # (B,C,H,W)
        wsum = torch.softmax(self.w, 0)
        return x * (wsum[0] * a1 + wsum[1] * a2 + wsum[2] * a3)


CUSTOM_36 = [C2f_iRMB, ThreeChAgg]           # 供注册
REPEAT_36 = [C2f_iRMB]                        # 带 repeat(n) 参数的
