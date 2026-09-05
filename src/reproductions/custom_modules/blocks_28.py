# -*- coding: utf-8 -*-
"""
[28]《YOLOV8-CMS 柑橘叶病分类与分级》自定义模块 —— 按论文文字描述**重建**。
⚠ 无官方代码。论文仅提到 CSPPC / MultiDimen / SpatialConv 三个名称, 细节未公开,
  这里做**保真度最低**的合理重建(CSP+PartialConv / 多维注意力 / 空间卷积), 仅供跑通与对照。
"""
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, C2f


class PConv(nn.Module):
    """Partial Convolution(FasterNet): 只对一部分通道做卷积, 其余恒等。"""
    def __init__(self, c, n_div=4):
        super().__init__()
        self.c1 = c // n_div
        self.conv = nn.Conv2d(self.c1, self.c1, 3, 1, 1, bias=False)
    def forward(self, x):
        x1, x2 = x[:, :self.c1], x[:, self.c1:]
        return torch.cat([self.conv(x1), x2], 1)


class PConvBlock(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.add = shortcut and c1 == c2
        self.pc = PConv(c1)
        self.pw1 = Conv(c1, c2, 1)
        self.pw2 = Conv(c2, c2, 1, act=False)
    def forward(self, x):
        y = self.pw2(self.pw1(self.pc(x)))
        return x + y if self.add else y


class CSPPC(C2f):
    """CSP + PartialConv 瓶颈(重建 [28] CSPPC)。签名与 C2f 一致。"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(PConvBlock(self.c, self.c, shortcut) for _ in range(n))


class MultiDimen(nn.Module):
    """多维注意力(重建 [28] MultiDimen): C-H / C-W / H-W 三维交互(triplet 思路), 通道保持。"""
    def __init__(self, c1, c2):
        super().__init__()
        assert c1 == c2, "MultiDimen 通道保持, 需 c1==c2"
        self.cw = nn.Conv2d(2, 1, 7, padding=3)
        self.ch = nn.Conv2d(2, 1, 7, padding=3)
        self.hw = nn.Conv2d(2, 1, 7, padding=3)
    @staticmethod
    def _att(x, conv):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return torch.sigmoid(conv(s))
    def forward(self, x):
        a_cw = self._att(x.permute(0, 2, 1, 3), self.cw).permute(0, 2, 1, 3)   # C-W
        a_ch = self._att(x.permute(0, 3, 2, 1), self.ch).permute(0, 3, 2, 1)   # C-H
        a_hw = self._att(x, self.hw)                                            # H-W
        return x * (a_cw + a_ch + a_hw) / 3.0


class SpatialConv(nn.Module):
    """空间卷积(重建 [28] SpatialConv): 深度可分离 + 空间门控, 可改变通道。签名 (c1,c2,k,s)。"""
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dw = Conv(c1, c1, k, s, g=c1)
        self.pw = Conv(c1, c2, 1, 1)
        self.gate = nn.Conv2d(c2, 1, 7, padding=3)
    def forward(self, x):
        y = self.pw(self.dw(x))
        return y * torch.sigmoid(self.gate(y))


CUSTOM_28 = [CSPPC, MultiDimen, SpatialConv]     # 供注册
REPEAT_28 = [CSPPC]                              # 带 repeat(n) 参数的
BASE_EXTRA_28 = [SpatialConv]                    # 改变通道、需 (c1,c2) 处理但无 repeat
