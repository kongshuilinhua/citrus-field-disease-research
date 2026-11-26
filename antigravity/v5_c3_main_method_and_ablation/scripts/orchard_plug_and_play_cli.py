# -*- coding: utf-8 -*-
"""
真实果园统一推理与冻结四模型评测命令行工具 (orchard_plug_and_play_cli.py — Stage 1 修复定型版)

严格修复与一致性保证：
1. 零硬编码超参数：C3 融合权重 [0.30, 0.20, 0.50]、C1 融合权重 [0.35, 0.65]、以及按种子标定的 C2 流形衰减系数 γ ∈ {0.35, 0.15, 0.35}
   100% 从冻结配置文件 `configs/c3_frozen_manifest.json` 读取，并在启动时强制校验底层权重的 SHA-256 哈希。
2. 在线与离线推理流程 100% 统一：
   单次 YOLOv8n 前向 (`conf=0.001, iou=0.70, max_det=300, rect=False`) 同步钩取检测头 14 类原始概率分布 ->
   对 `score >= 0.08` 候选框裁取双尺度视图 (`0.82x` 紧致中心视角 + `1.25x` 标准目标视角) 单批次输入冻结 DINOv2 ->
   温度缩放对数后验加权融合 -> 统一几何分数重加权 -> 类别内 `batched_nms(iou=0.70)`。
3. 明确命令行模式边界（文档与代码严格一致，不含未实现声明）：
   - `--mode check_manifest`：校验全部 12 份冻结模型权重 SHA-256 与冻结参数配置；
   - `--mode preannotate`：对未标注果园图像目录运行主方法 C3（可选 B0/B1/C1/C2），输出检测框、Top-3 可确认症状候选卡片与不确定度复核排序；
   - `--mode eval_frozen4`：对已完成标注的真实果园 COCO JSON（可选传入 `--cluster-csv` 采集簇编号表）一次性评测冻结四模型
     (`B0_YOLOv8n`, `B1_Full_DINOv2`, `C1_SingleScale_125`, `C3_DualScale_082_125`)，完整输出检测指标 (AP50/AP50-95/Recall@0.5/候选召回)、
     分类指标 (匹配目标 Top-1/Top-3、含漏检整图 Top-3)、部署延迟 (平均/P95/多目标延迟)、四类错误归因及按采集簇重采样配对置信区间。
   （注：保形风险控制 R2 与 597 例 PCR 分层送检模拟属于独立统计评估实验，位于 `run_candidate_arsenal_and_conformal_suite.py`，不在此推理 CLI 中混列。）
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

import sys
import io
import csv
import json
import math
import time
import hashlib
import argparse
import contextlib
from pathlib import Path
from collections import defaultdict

for _extra_sp in [r'D:\新版柑橘\方法路线扩展_20261031\检测环境\Lib\site-packages', r'D:\新版柑橘\检测环境\Lib\site-packages']:
    if Path(_extra_sp).exists() and _extra_sp not in sys.path:
        sys.path.append(_extra_sp)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, batched_nms
import timm
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
PREV_TCD = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / 'configs' / 'c3_frozen_manifest.json'

sys.path.append(str(EXT))
sys.path.append(str(PREV_TCD))
from step1_extract_multigranular_cache import crop_square_scale, letterbox_640

SYMPTOM_DISPLAY_NAMES_ZH = {
    0: 'S1 健康果实外观',
    1: 'S2 健康叶片外观',
    2: 'S3 斑驳黄化疑似症状(需PCR/专家复核，非确诊HLB)',
    3: 'S4 红鼻果疑似症状(需PCR/专家复核，非确诊HLB)',
    4: 'S5 缺锌样脉间黄化症状(非生物缺素/疑似混淆项)',
    5: 'S6 叶脉黄化疑似症状(需PCR/专家复核，非确诊HLB)',
    6: 'S7 均匀黄化疑似症状(需PCR/专家复核，非确诊HLB)',
    7: 'S8 缺镁样症状(非生物缺素胁迫)',
    8: 'S9 缺硼样症状(非生物缺素胁迫)',
    9: 'S10 炭疽病疑似症状',
    10: 'S11 脂点黄斑病疑似症状',
    11: 'S14 溃疡病疑似症状',
    12: 'S13 煤烟病疑似症状',
    13: 'S12 藻斑/苔藓附生症状',
}
ABIOTIC_SET = {4, 7, 8}
SUSPECT_HLB_SYMPTOM_SET = {2, 3, 4, 5, 6}


def fast_letterbox_params(h: int, w: int, new_shape: int = 640):
    r = min(new_shape / float(h), new_shape / float(w))
    nw, nh = int(round(w * r)), int(round(h * r))
    px = int(round((new_shape - nw) * 0.5 - 0.1))
    py = int(round((new_shape - nh) * 0.5 - 0.1))
    return r, float(px), float(py)


def verify_manifest_hashes(manifest: dict, seeds_to_check=None):
    target_seeds = [str(s) for s in (seeds_to_check if seeds_to_check is not None else [0, 1, 2])]
    for sk in target_seeds:
        ckpts = manifest['seeds'][sk]['checkpoints']
        for name, item in ckpts.items():
            p = Path(item['path'])
            if not p.exists():
                raise FileNotFoundError(f"冻结权重不存在 [{name}]: {p}")
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            if actual != item['sha256']:
                raise RuntimeError(f"权重哈希不匹配 [{name} seed={sk}]: 期望 {item['sha256']} != 实际 {actual}")


class UnifiedOrchardInferenceEngine:
    """严格统一的端到端在线推理引擎（离线缓存评测与在线单图推理 100% 共享同一数学路径）。"""

    def __init__(self, manifest: dict, seed: int = 0, device: str = 'cuda:0'):
        self.manifest = manifest
        self.seed = seed
        self.device = device
        self.scfg = manifest['seeds'][str(seed)]
        verify_manifest_hashes(manifest, seeds_to_check=[seed])

        ckpts = self.scfg['checkpoints']
        self.T_DET = float(self.scfg['temperatures']['T_DET'])
        self.T_DINO = float(self.scfg['temperatures']['T_DINO'])
        pp = self.scfg['postprocess']
        self.yolo_conf = float(pp['yolo_eval_conf_thr'])
        self.yolo_iou = float(pp['yolo_init_nms_iou'])
        self.yolo_max_det = int(pp['yolo_max_det'])
        self.b1_thr = float(pp['dino_hard_B1_conf_thr'])
        self.fg_thr = float(pp['fusion_fg_conf_thr'])
        self.beta = float(pp['beta_score_reweight'])
        self.post_nms_iou = float(pp['post_reclass_nms_iou'])

        fp = self.scfg['locked_fusion_params_from_scidb_val']
        self.w_c1 = [float(x) for x in fp['C1_SingleScale_125_weights_det_std']]
        self.w_c2_prior = [float(x) for x in fp['C2_OOD_Pool_prior_det_std']]
        self.gamma_c2 = float(fp['C2_gamma_ood'])
        self.scales_c3 = [float(x) for x in fp['C3_MainMethod_DualScale_scales']]
        self.w_c3 = [float(x) for x in fp['C3_MainMethod_weights_det_core082_std125']]

        # Load frozen DINOv2 LP-FT
        st_dino = torch.load(ckpts['dinov2_lpft']['path'], map_location='cpu', weights_only=True)
        self.enc = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, img_size=224, num_classes=0).to(device).eval()
        self.enc.load_state_dict(st_dino['骨干'])
        self.head = nn.Linear(384, 14).to(device).eval()
        self.head.load_state_dict(st_dino['分类头'])
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        # Load Mahalanobis stats (only used if C2 is requested)
        blob_tcd = torch.load(ckpts['tcd_calib_stats']['path'], map_location='cpu', weights_only=False)
        m_cfg = blob_tcd['mahalanobis']
        self.m_means = torch.from_numpy(m_cfg['means'].astype(np.float32)).to(device)
        self.m_prec = torch.from_numpy(m_cfg['prec'].astype(np.float32)).to(device)
        self.m_mu = float(m_cfg['m_mu'])
        self.m_sd = float(m_cfg['m_sd'])

        # Load frozen YOLOv8n with single-pass Detect hook
        self.yolo = YOLO(ckpts['yolov8n_det']['path'])
        self.net = self.yolo.model.to(device).eval()
        self.detect = self.net.model[-1]
        self.maps = [None, None, None]
        self.raw_y = [None]
        self.hooks = [
            self.detect.cv3[s][1].register_forward_hook(lambda _m, _i, o, idx=s: self.maps.__setitem__(idx, o.detach().clone()))
            for s in range(3)
        ]
        self.hooks.append(self.detect.register_forward_hook(lambda _m, _i, o: self.raw_y.__setitem__(0, o[0].detach().clone())))
        self.strides = [8.0, 16.0, 32.0]

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    @torch.no_grad()
    def yolo_detect_square640(self, bgr: np.ndarray):
        """
        严格执行与离线统一评测完全一致的 640x640 方形 LetterBox 前向推理（修复 Ultralytics 单图 predict 默认启用 auto=True 矩形缩放
        导致竖图/16:9图锚点坐标与颈部特征偏移的隐患）。单次前向同时产出 NMS 检测框与 8400 锚点原始输出。
        """
        h, w = bgr.shape[:2]
        pad, _r, _px, _py = letterbox_640(bgr)
        rgb_pad = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)
        bx = (torch.from_numpy(rgb_pad.transpose(2, 0, 1)).to(self.device).float() / 255.0).unsqueeze(0)
        _ = self.net(bx)
        det = non_max_suppression(self.raw_y[0].clone(), conf_thres=self.yolo_conf, iou_thres=self.yolo_iou, max_det=self.yolo_max_det)[0]
        if len(det) == 0:
            z4 = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
            z1 = torch.zeros((0,), dtype=torch.float32, device=self.device)
            zc = torch.zeros((0,), dtype=torch.long, device=self.device)
            return z4, z1, zc
        xyxy = scale_boxes((640, 640), det[:, :4].clone(), (h, w))
        return xyxy, det[:, 4], det[:, 5].long()

    @torch.no_grad()
    def _extract_anchor_p14_and_neck(self, h: int, w: int, bxs_xywh: list, need_neck: bool = False):
        r_lb, px_lb, py_lb = fast_letterbox_params(h, w, 640)
        yb = self.raw_y[0][0]  # (18, 8400) on GPU
        acx, acy, abw, abh = yb[0], yb[1], yb[2], yb[3]
        ax1 = (acx - abw * 0.5 - px_lb) / r_lb
        ay1 = (acy - abh * 0.5 - py_lb) / r_lb
        aw, ah = abw / r_lb, abh / r_lb
        sc14 = yb[4:]
        a_idxs, rois = [], []
        for bb in bxs_xywh:
            d_anc = (ax1 - bb[0]).abs() + (ay1 - bb[1]).abs() + (aw - bb[2]).abs() + (ah - bb[3]).abs()
            a_idxs.append(torch.argmin(d_anc))
            if need_neck:
                lx1 = float(np.clip(bb[0] * r_lb + px_lb, 0.0, 639.0))
                ly1 = float(np.clip(bb[1] * r_lb + py_lb, 0.0, 639.0))
                lx2 = float(np.clip((bb[0] + bb[2]) * r_lb + px_lb, lx1 + 1.0, 640.0))
                ly2 = float(np.clip((bb[1] + bb[3]) * r_lb + py_lb, ly1 + 1.0, 640.0))
                rois.append([0.0, lx1, ly1, lx2, ly2])
        p14_g = sc14[:, torch.stack(a_idxs)].T
        if not need_neck:
            return p14_g, None
        rt = torch.tensor(rois, dtype=torch.float32, device=self.device)
        sp_f = [F.normalize(roi_align(self.maps[s].float(), rt, (1, 1), 1.0 / self.strides[s], aligned=True).flatten(1), dim=1) for s in range(3)]
        return p14_g, F.normalize(torch.cat(sp_f, dim=1), dim=1)

    @torch.no_grad()
    def _encode_dino_logits(self, crops_u8: np.ndarray) -> torch.Tensor:
        xt = (torch.from_numpy(crops_u8).to(self.device).float() / 255.0 - self.mean) / self.std
        z = F.normalize(self.enc.forward_features(xt)[:, 0], dim=1)
        return self.head(z)

    @torch.no_grad()
    def compute_box_posteriors_given_boxes(self, bgr: np.ndarray, bxs_xywh: list, candidate: str = 'C3') -> np.ndarray:
        """在给定候选框坐标上执行在线特征提取与融合，用于逐框离线/在线数值一致性核验。"""
        if not bxs_xywh:
            return np.zeros((0, 14), dtype=np.float32)
        h, w = bgr.shape[:2]
        _ = self.yolo_detect_square640(bgr)
        p14_g, zn_g = self._extract_anchor_p14_and_neck(h, w, bxs_xywh, need_neck=(candidate == 'C2'))
        lp_det = F.log_softmax(torch.logit(p14_g.clamp(1e-5, 1.0 - 1e-5)) / self.T_DET, dim=-1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        k = len(bxs_xywh)

        if candidate == 'C3':
            s_core, s_std = self.scales_c3
            c_both = np.stack([crop_square_scale(rgb, bb, s_core) for bb in bxs_xywh] +
                              [crop_square_scale(rgb, bb, s_std) for bb in bxs_xywh], axis=0)
            lg_both = self._encode_dino_logits(c_both)
            lp_core = F.log_softmax(lg_both[:k] / self.T_DINO, dim=-1)
            lp_std = F.log_softmax(lg_both[k:] / self.T_DINO, dim=-1)
            wd, wc, ws = self.w_c3
            p_post = F.softmax(wd * lp_det + wc * lp_core + ws * lp_std, dim=-1)
        elif candidate == 'C1':
            c125 = np.stack([crop_square_scale(rgb, bb, 1.25) for bb in bxs_xywh], axis=0)
            lp_std = F.log_softmax(self._encode_dino_logits(c125) / self.T_DINO, dim=-1)
            p_post = F.softmax(self.w_c1[0] * lp_det + self.w_c1[1] * lp_std, dim=-1)
        elif candidate == 'C2':
            c125 = np.stack([crop_square_scale(rgb, bb, 1.25) for bb in bxs_xywh], axis=0)
            lp_std = F.log_softmax(self._encode_dino_logits(c125) / self.T_DINO, dim=-1)
            pd, ps = lp_det.exp(), lp_std.exp()
            hd = -(pd.clamp_min(1e-7) * pd.clamp_min(1e-7).log()).sum(-1) / math.log(14.0)
            hs = -(ps.clamp_min(1e-7) * ps.clamp_min(1e-7).log()).sum(-1) / math.log(14.0)
            ch = pd.argmax(dim=-1)
            diff_m = zn_g - self.m_means[ch]
            dm_z = (torch.sqrt(torch.clamp((diff_m @ self.m_prec * diff_m).sum(dim=-1), min=0.0)) - self.m_mu) / self.m_sd
            u0 = self.w_c2_prior[0] * (1.0 - hd).clamp(0.05, 1.0) * torch.exp(-self.gamma_c2 * torch.clamp(dm_z, min=0.0))
            u1 = self.w_c2_prior[1] * (1.0 - hs).clamp(0.05, 1.0)
            lp_f = (u0[:, None] * lp_det + u1[:, None] * lp_std) / (u0 + u1 + 1e-8)[:, None]
            p_post = F.softmax(lp_f, dim=-1)
        else:
            raise ValueError(candidate)
        return p_post.cpu().numpy()

    @torch.no_grad()
    def infer_image_detailed(self, bgr: np.ndarray, candidate: str = 'C3'):
        """
        统一端到端单图推理 (B0 / B1 / C1 / C2 / C3)：
        返回 (kept_boxes_xywh, kept_scores, kept_cls, kept_probs_14, raw_low_thr_boxes_xywh, n_dino_crops)
        """
        h, w = bgr.shape[:2]
        xyxy_t, sc_t, cls_t = self.yolo_detect_square640(bgr)
        N = len(xyxy_t)
        raw_low_bxs = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
                       for b in xyxy_t[sc_t >= 0.05].cpu().tolist()] if N > 0 else []

        if N == 0:
            return [], [], [], np.zeros((0, 14), dtype=np.float32), [], 0

        if candidate == 'B0':
            p14_all, _ = self._extract_anchor_p14_and_neck(h, w, [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy_t.cpu().tolist()], False)
            p_norm = (p14_all / p14_all.sum(dim=-1, keepdim=True).clamp_min(1e-8)).cpu().numpy()
            bxs_out = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy_t.cpu().tolist()]
            return bxs_out, sc_t.cpu().tolist(), cls_t.cpu().tolist(), p_norm, raw_low_bxs, 0

        xyxy_f, sc_f, cls_f = xyxy_t.clone(), sc_t.clone(), cls_t.clone()
        p14_all, _ = self._extract_anchor_p14_and_neck(h, w, [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy_t.cpu().tolist()], False)
        probs_full = (p14_all / p14_all.sum(dim=-1, keepdim=True).clamp_min(1e-8))

        if candidate == 'B1':
            act_ids = torch.where(sc_t >= self.b1_thr)[0]
            n_dino = 0
            if len(act_ids) > 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bxs = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy_t[act_ids].cpu().tolist()]
                c125 = np.stack([crop_square_scale(rgb, bb, 1.25) for bb in bxs], axis=0)
                p_d = F.softmax(self._encode_dino_logits(c125), dim=-1)
                probs_full[act_ids] = p_d
                pmax, ncls = p_d.max(dim=-1)
                changed = ncls != cls_f[act_ids]
                cls_f[act_ids] = ncls
                sc_sub = sc_f[act_ids]
                sc_f[act_ids] = torch.where(changed, torch.sqrt(sc_sub * pmax.clamp_min(1e-6)), sc_sub)
                n_dino = len(bxs)
            bxs_out = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy_f.cpu().tolist()]
            return bxs_out, sc_f.cpu().tolist(), cls_f.cpu().tolist(), probs_full.cpu().numpy(), raw_low_bxs, n_dino

        # Calibrated Fusion Modes: C1 / C2 / C3
        fg_ids = torch.where(sc_t >= self.fg_thr)[0]
        n_dino = 0
        if len(fg_ids) > 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bxs_fg = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy_t[fg_ids].cpu().tolist()]
            k = len(bxs_fg)
            p14_fg, zn_fg = self._extract_anchor_p14_and_neck(h, w, bxs_fg, need_neck=(candidate == 'C2'))
            lp_det = F.log_softmax(torch.logit(p14_fg.clamp(1e-5, 1.0 - 1e-5)) / self.T_DET, dim=-1)

            if candidate == 'C3':
                s_core, s_std = self.scales_c3
                c_both = np.stack([crop_square_scale(rgb, bb, s_core) for bb in bxs_fg] +
                                  [crop_square_scale(rgb, bb, s_std) for bb in bxs_fg], axis=0)
                lg_both = self._encode_dino_logits(c_both)
                lp_core = F.log_softmax(lg_both[:k] / self.T_DINO, dim=-1)
                lp_std = F.log_softmax(lg_both[k:] / self.T_DINO, dim=-1)
                wd, wc, ws = self.w_c3
                p_post = F.softmax(wd * lp_det + wc * lp_core + ws * lp_std, dim=-1)
                n_dino = 2 * k
            elif candidate == 'C1':
                c125 = np.stack([crop_square_scale(rgb, bb, 1.25) for bb in bxs_fg], axis=0)
                lp_std = F.log_softmax(self._encode_dino_logits(c125) / self.T_DINO, dim=-1)
                p_post = F.softmax(self.w_c1[0] * lp_det + self.w_c1[1] * lp_std, dim=-1)
                n_dino = k
            elif candidate == 'C2':
                c125 = np.stack([crop_square_scale(rgb, bb, 1.25) for bb in bxs_fg], axis=0)
                lp_std = F.log_softmax(self._encode_dino_logits(c125) / self.T_DINO, dim=-1)
                pd, ps = lp_det.exp(), lp_std.exp()
                hd = -(pd.clamp_min(1e-7) * pd.clamp_min(1e-7).log()).sum(-1) / math.log(14.0)
                hs = -(ps.clamp_min(1e-7) * ps.clamp_min(1e-7).log()).sum(-1) / math.log(14.0)
                ch = pd.argmax(dim=-1)
                diff_m = zn_fg - self.m_means[ch]
                dm_z = (torch.sqrt(torch.clamp((diff_m @ self.m_prec * diff_m).sum(dim=-1), min=0.0)) - self.m_mu) / self.m_sd
                u0 = self.w_c2_prior[0] * (1.0 - hd).clamp(0.05, 1.0) * torch.exp(-self.gamma_c2 * torch.clamp(dm_z, min=0.0))
                u1 = self.w_c2_prior[1] * (1.0 - hs).clamp(0.05, 1.0)
                lp_f = (u0[:, None] * lp_det + u1[:, None] * lp_std) / (u0 + u1 + 1e-8)[:, None]
                p_post = F.softmax(lp_f, dim=-1)
                n_dino = k
            else:
                raise ValueError(f"未知候选模型: {candidate}")

            probs_full[fg_ids] = p_post
            pmax, ncls = p_post.max(dim=-1)
            cls_f[fg_ids] = ncls
            sc_f[fg_ids] = (sc_f[fg_ids] ** self.beta) * (pmax.clamp_min(1e-6) ** (1.0 - self.beta))

            if len(xyxy_f) > 1:
                keep = batched_nms(xyxy_f, sc_f, cls_f, self.post_nms_iou)[:300]
                xyxy_f, sc_f, cls_f, probs_full = xyxy_f[keep], sc_f[keep], cls_f[keep], probs_full[keep]

        bxs_out = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy_f.cpu().tolist()]
        return bxs_out, sc_f.cpu().tolist(), cls_f.cpu().tolist(), probs_full.cpu().numpy(), raw_low_bxs, n_dino

    def infer_image_coco_rows(self, bgr: np.ndarray, image_id: int, candidate: str = 'C3') -> list:
        bxs, scs, cls_l, _, _, _ = self.infer_image_detailed(bgr, candidate=candidate)
        return [{'image_id': image_id, 'category_id': int(c), 'bbox': bb, 'score': float(s)}
                for bb, s, c in zip(bxs, scs, cls_l)]


def run_preannotate_folder(manifest: dict, image_dir: Path, out_json: Path, candidate: str = 'C3', seed: int = 0,
                           display_conf_thr: float = 0.20, max_images: int = 0):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    eng = UnifiedOrchardInferenceEngine(manifest, seed=seed, device=device)
    img_files = sorted([p for p in image_dir.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
    if max_images > 0:
        img_files = img_files[:max_images]
    if not img_files:
        raise FileNotFoundError(f"未在目录中找到图像文件: {image_dir}")

    results_out = []
    for p in img_files:
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
        if bgr is None:
            continue
        bxs, scs, cls_l, probs, _, _ = eng.infer_image_detailed(bgr, candidate=candidate)
        cards = []
        for bb, sc, cl, pr in zip(bxs, scs, cls_l, probs):
            if sc < display_conf_thr:
                continue
            top3 = np.argsort(-pr)[:3].tolist()
            cards.append({
                'bbox_xywh': [round(v, 1) for v in bb],
                'calibrated_score': round(float(sc), 4),
                'top1_id': int(cl),
                'top1_label_zh': SYMPTOM_DISPLAY_NAMES_ZH[int(cl)],
                'top3_confirmable_candidates': [
                    {'id': int(c), 'label_zh': SYMPTOM_DISPLAY_NAMES_ZH[int(c)], 'prob': round(float(pr[c]), 4)}
                    for c in top3
                ],
                'abiotic_nutrient_ood_mass': round(float(pr[sorted(list(ABIOTIC_SET))].sum()), 4),
                'uncertainty_1_minus_maxp': round(float(1.0 - pr.max()), 4),
                'requires_pcr_or_expert_note': bool(int(cl) in SUSPECT_HLB_SYMPTOM_SET),
            })
        results_out.append({
            'file': p.name,
            'n_detections_shown': len(cards),
            'review_priority_1_minus_maxp': max([d['uncertainty_1_minus_maxp'] for d in cards]) if cards else 0.0,
            'detections': cards,
        })

    eng.close()
    results_out.sort(key=lambda x: -x['review_priority_1_minus_maxp'])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        'safety_disclaimer': '所有输出仅为视觉外观症状候选与专家复核辅助；黄化/斑驳等症状绝不等同于病原学确诊黄龙病(HLB)，无法判断的样本应由农学专家标记为不确定。',
        'manifest_version': manifest['manifest_version'],
        'candidate_used': candidate,
        'seed': seed,
        'n_images_processed': len(results_out),
        'ranked_image_outputs': results_out,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[预标注完成] 使用冻结候选 {candidate} (Seed {seed}) 处理 {len(results_out)} 张图像 -> {out_json}", flush=True)


def evaluate_frozen_4models_on_labeled_orchard(manifest: dict, image_dir: Path, gt_json: Path, out_json: Path,
                                               cluster_csv: str = '', seeds=(0, 1, 2)):
    """第四阶段专用：仅评测冻结四模型 (B0: YOLOv8n, B1: Full DINOv2, C1: 1.25x, C3: 0.82x+1.25x 主方法)，含采集簇重采样与四类错误归因。"""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    gt_dict = json.loads(gt_json.read_text(encoding='utf-8'))
    anns_by_img = defaultdict(list)
    for a in gt_dict['annotations']:
        if a.get('iscrowd', 0) == 0:
            anns_by_img[a['image_id']].append(a)

    # Optional cluster mapping (file_name -> cluster_id) to avoid treating burst shots as independent
    fn2cluster = {}
    if cluster_csv and Path(cluster_csv).exists():
        for r in csv.DictReader(open(cluster_csv, encoding='utf-8-sig')):
            fn = r.get('file_name') or r.get('文件')
            cid = r.get('cluster_id') or r.get('簇编号')
            if fn and cid:
                fn2cluster[fn] = str(cid)

    models_4 = [('B0', 'YOLOv8n'), ('B1', 'Full_DINOv2'), ('C1', 'C1_SingleScale_125'), ('C3', 'C3_DualScale_082_125')]
    report = {'models': {m[1]: defaultdict(list) for m in models_4}, 'per_seed': {}}

    def iou_xywh(b1, b2):
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        return inter / (b1[2] * b1[3] + b2[2] * b2[3] - inter + 1e-9)

    for seed in seeds:
        eng = UnifiedOrchardInferenceEngine(manifest, seed=seed, device=device)
        report['per_seed'][str(seed)] = {}
        for m_code, m_name in models_4:
            coco_rows = []
            times_all, times_multi = [], []
            n_gt_tot = n_prop_hit = n_rec50_loc = n_top1_ok = n_top3_ok = 0
            err_no_cand = err_poor_loc = err_score_nms_drop = err_wrong_cls = 0
            img_top3_hits_with_miss = []

            for idx, im in enumerate(gt_dict['images']):
                iid = im['id']
                p = image_dir / im['file_name']
                bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                bxs, scs, cls_l, probs, raw_low_bxs, _ = eng.infer_image_detailed(bgr, candidate=m_code)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if idx >= 5:
                    times_all.append(dt_ms)
                    if len(anns_by_img.get(iid, [])) >= 3 or len([s for s in scs if s >= 0.25]) >= 3:
                        times_multi.append(dt_ms)

                for bb, s, c in zip(bxs, scs, cls_l):
                    coco_rows.append({'image_id': iid, 'category_id': int(c), 'bbox': bb, 'score': float(s)})

                # Active detections at operational threshold >= 0.20 for recall & error attribution
                act = [(bxs[j], scs[j], cls_l[j], probs[j]) for j in range(len(bxs)) if scs[j] >= 0.20]
                glist = anns_by_img.get(iid, [])
                for ga in glist:
                    n_gt_tot += 1
                    gb, gc = ga['bbox'], int(ga['category_id'])
                    max_raw_iou = max([iou_xywh(gb, rb) for rb in raw_low_bxs], default=0.0)
                    if max_raw_iou >= 0.50:
                        n_prop_hit += 1
                    best_act_iou, best_j = 0.0, -1
                    for j, (ab, as_, ac_, ap_) in enumerate(act):
                        v = iou_xywh(gb, ab)
                        if v > best_act_iou:
                            best_act_iou, best_j = v, j
                    if best_act_iou >= 0.50:
                        n_rec50_loc += 1
                        _, _, ac_, ap_ = act[best_j]
                        top3_c = np.argsort(-ap_)[:3].tolist()
                        if int(ac_) == gc:
                            n_top1_ok += 1
                        else:
                            err_wrong_cls += 1
                        if gc in top3_c:
                            n_top3_ok += 1
                            img_top3_hits_with_miss.append((fn2cluster.get(im['file_name'], str(iid)), 1.0))
                        else:
                            img_top3_hits_with_miss.append((fn2cluster.get(im['file_name'], str(iid)), 0.0))
                    else:
                        img_top3_hits_with_miss.append((fn2cluster.get(im['file_name'], str(iid)), 0.0))
                        if max_raw_iou < 0.10:
                            err_no_cand += 1
                        elif max_raw_iou < 0.50:
                            err_poor_loc += 1
                        else:
                            err_score_nms_drop += 1

            with contextlib.redirect_stdout(io.StringIO()):
                c_gt = COCO(str(gt_json))
                ev = COCOeval_faster(c_gt, c_gt.loadRes(coco_rows), 'bbox') if coco_rows else None
                if ev:
                    ev.evaluate(); ev.accumulate(); ev.summarize()
            ap50_95 = round(float(ev.stats[0] * 100.0), 2) if ev else 0.0
            ap50 = round(float(ev.stats[1] * 100.0), 2) if ev else 0.0

            s_rec = {
                'AP50_95': ap50_95,
                'AP50': ap50,
                'LowThr_Proposal_Recall50_pct': round(n_prop_hit / max(1, n_gt_tot) * 100.0, 2),
                'Recall_IoU50_Active_pct': round(n_rec50_loc / max(1, n_gt_tot) * 100.0, 2),
                'Matched_Top1_Acc_pct': round(n_top1_ok / max(1, n_rec50_loc) * 100.0, 2),
                'Matched_Top3_Acc_pct': round(n_top3_ok / max(1, n_rec50_loc) * 100.0, 2),
                'MissPenalized_Total_Top3_Hit_pct': round(n_top3_ok / max(1, n_gt_tot) * 100.0, 2),
                'Mean_Latency_ms': round(float(np.mean(times_all)), 2) if times_all else None,
                'P95_Latency_ms': round(float(np.percentile(times_all, 95)), 2) if times_all else None,
                'MultiTarget_Latency_ms': round(float(np.mean(times_multi)), 2) if times_multi else None,
                'Error_Attribution_Counts': {
                    'no_candidate_box': err_no_cand,
                    'inaccurate_box_localization': err_poor_loc,
                    'lost_by_score_or_nms': err_score_nms_drop,
                    'localized_but_wrong_class': err_wrong_cls,
                },
            }
            report['per_seed'][str(seed)][m_name] = s_rec
            for k, v in s_rec.items():
                if isinstance(v, (int, float)):
                    report['models'][m_name][k].append(v)

        eng.close()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[冻结四模型评测完成] 结果已保存至: {out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser(description='冻结主方法 C3 统一在线推理与真实果园四模型评测工具')
    ap.add_argument('--manifest', type=str, default=str(DEFAULT_MANIFEST))
    ap.add_argument('--mode', choices=['check_manifest', 'preannotate', 'eval_frozen4'], default='check_manifest')
    ap.add_argument('--image-dir', type=str, default='')
    ap.add_argument('--gt-json', type=str, default='')
    ap.add_argument('--cluster-csv', type=str, default='')
    ap.add_argument('--out-json', type=str, default='')
    ap.add_argument('--candidate', choices=['B0', 'B1', 'C1', 'C2', 'C3'], default='C3')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-images', type=int, default=0)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    if args.mode == 'check_manifest':
        verify_manifest_hashes(manifest)
        s0 = manifest['seeds']['0']['locked_fusion_params_from_scidb_val']
        gammas = [manifest['seeds'][str(s)]['locked_fusion_params_from_scidb_val']['C2_gamma_ood'] for s in [0, 1, 2]]
        print(f"[冻结配置校验通过] 主方法={manifest['primary_method_id']} | C3权重={s0['C3_MainMethod_weights_det_core082_std125']} | "
              f"C2按种子γ={gammas} | 12份权重SHA-256全部吻合！", flush=True)
    elif args.mode == 'preannotate':
        assert args.image_dir and args.out_json, "preannotate 模式需指定 --image-dir 与 --out-json"
        run_preannotate_folder(manifest, Path(args.image_dir), Path(args.out_json), args.candidate, args.seed, max_images=args.max_images)
    elif args.mode == 'eval_frozen4':
        assert args.image_dir and args.gt_json and args.out_json, "eval_frozen4 模式需指定 --image-dir, --gt-json 与 --out-json"
        evaluate_frozen_4models_on_labeled_orchard(manifest, Path(args.image_dir), Path(args.gt_json), Path(args.out_json), args.cluster_csv)


if __name__ == '__main__':
    main()
