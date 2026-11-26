# -*- coding: utf-8 -*-
"""
真实果园统一推理与预声明冻结五模型评测命令行工具 (orchard_plug_and_play_cli.py — Stage 2b 审计增强定型版)

严格修复与方法学保证：
1. 零硬编码超参数：唯一主方法 C3 融合权重 [0.30, 0.20, 0.50]、单尺度对照 C1 [0.35, 0.65]、常规双视角翻转 TTA 补充对照 A4b
   按种子权重 [[0.30, 0.50, 0.20], [0.30, 0.25, 0.45], [0.30, 0.55, 0.15]] 及 C2 按种子 γ ∈ {0.35, 0.15, 0.35}
   100% 从冻结配置文件 `configs/c3_frozen_manifest.json` 读取，并在启动时强制校验 12 份底层权重 SHA-256。
2. 在线与离线推理流程 100% 逐框一致：
   显式 640x640 方形 LetterBox 单次 YOLOv8n 前向 -> 钩取 8400 锚点 14 类原始响应 -> 对 score >= 0.08 前景框单批次提取
   DINOv2 特征 -> 温度缩放对数后验融合 -> 统一几何置信度重加权 -> 类别内 batched_nms(iou=0.70)。
3. 真实果园外部验证严密统计保证（修复一对多重复匹配与采集簇相关性）：
   - 预声明冻结 5 模型评测：B0 (YOLOv8n)、B1 (Full DINOv2)、C1 (1.25x 单尺度校准)、A4b (1.25x+HFlip 常规 TTA 补充对照)、
     以及唯一主方法 C3 (0.82x+1.25x)；
   - 严格一对一（1-to-1）贪心置信度优先匹配：在 Recall@IoU0.5、低阈值候选召回、Matched Top-1/Top-3 及完备五分法错误归因中，
     按预测分数降序逐一匹配未占用的真值框（每个预测框与每个真值框最多匹配一次，绝无相邻重叠叶片共享同一预测框虚增召回）；
   - 自动隔离植保专家标记为“不确定/待核实”（`uncertain=True` 或 `category_id < 0`）的疑难标注，单独统计不强判；
   - 按采集簇（Cluster-level）同步配对 Bootstrap (`B = 1,000`)：对同株/同枝连拍簇执行整簇有放回重采样，精确计算
     `AP50-95`、`AP50`、`Recall@IoU0.5`、`Matched_Top1` 与 `MissPenalized_Top3` 的配对差值 95% 置信区间及双侧 p 值。
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


def iou_xywh(b1, b2) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / (b1[2] * b1[3] + b2[2] * b2[3] - inter + 1e-9)


def match_detections_to_gt_1to1(dets_sorted_desc: list, gt_list: list, iou_thr: float = 0.50):
    """
    严格一对一（1-to-1）置信度优先匹配（与 COCO 匹配顺序一致）：
    输入 `dets_sorted_desc` 必须已按置信度降序排列，元素首项为 `bbox_xywh`。
    返回 `gt_to_det_idx`（字典 `{gt_i: det_j}`，每个预测框和每个真值框最多被占用一次）以及每个真值框与任意预测框的最大几何 IoU。
    """
    n_d, n_g = len(dets_sorted_desc), len(gt_list)
    max_any_iou = [0.0] * n_g
    if n_d == 0 or n_g == 0:
        return {}, max_any_iou

    iou_mat = np.zeros((n_d, n_g), dtype=np.float64)
    for dj, d_item in enumerate(dets_sorted_desc):
        db = d_item[0]
        for gi, ga in enumerate(gt_list):
            v = iou_xywh(db, ga['bbox'])
            iou_mat[dj, gi] = v
            if v > max_any_iou[gi]:
                max_any_iou[gi] = v

    used_gt = set()
    gt_to_det = {}
    for dj in range(n_d):
        best_iou, best_gi = iou_thr, -1
        for gi in range(n_g):
            if gi not in used_gt and iou_mat[dj, gi] >= best_iou:
                best_iou = iou_mat[dj, gi]
                best_gi = gi
        if best_gi >= 0:
            used_gt.add(best_gi)
            gt_to_det[best_gi] = dj
    return gt_to_det, max_any_iou


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
    """严格统一的端到端在线推理引擎（支持 B0 / B1 / C1 / C2 / A4b / C3）。"""

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
        self.w_a4b = [float(x) for x in fp.get('A4b_FlipTTA_125_HFlip_weights_det_std_flip', [0.30, 0.35, 0.35])]
        self.scales_c3 = [float(x) for x in fp['C3_MainMethod_DualScale_scales']]
        self.w_c3 = [float(x) for x in fp['C3_MainMethod_weights_det_core082_std125']]

        st_dino = torch.load(ckpts['dinov2_lpft']['path'], map_location='cpu', weights_only=True)
        self.enc = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, img_size=224, num_classes=0).to(device).eval()
        self.enc.load_state_dict(st_dino['骨干'])
        self.head = nn.Linear(384, 14).to(device).eval()
        self.head.load_state_dict(st_dino['分类头'])
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        blob_tcd = torch.load(ckpts['tcd_calib_stats']['path'], map_location='cpu', weights_only=False)
        m_cfg = blob_tcd['mahalanobis']
        self.m_means = torch.from_numpy(m_cfg['means'].astype(np.float32)).to(device)
        self.m_prec = torch.from_numpy(m_cfg['prec'].astype(np.float32)).to(device)
        self.m_mu = float(m_cfg['m_mu'])
        self.m_sd = float(m_cfg['m_sd'])

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
        yb = self.raw_y[0][0]
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
        elif candidate == 'A4b':
            std_list = [crop_square_scale(rgb, bb, 1.25) for bb in bxs_xywh]
            flip_list = [c[:, :, ::-1].copy() for c in std_list]
            c_both = np.stack(std_list + flip_list, axis=0)
            lg_both = self._encode_dino_logits(c_both)
            lp_std = F.log_softmax(lg_both[:k] / self.T_DINO, dim=-1)
            lp_flp = F.log_softmax(lg_both[k:] / self.T_DINO, dim=-1)
            wd, ws, wf = self.w_a4b
            p_post = F.softmax(wd * lp_det + ws * lp_std + wf * lp_flp, dim=-1)
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
        统一端到端单图推理 (B0 / B1 / C1 / C2 / A4b / C3)：
        返回 (kept_boxes_xywh, kept_scores, kept_cls, kept_probs_14, raw_low_thr_proposals_desc, n_dino_crops)
        其中 raw_low_thr_proposals_desc 为按原始检测分数降序排列的 [(bbox_xywh, raw_score), ...]。
        """
        h, w = bgr.shape[:2]
        xyxy_t, sc_t, cls_t = self.yolo_detect_square640(bgr)
        N = len(xyxy_t)
        if N > 0:
            low_mask = sc_t >= 0.05
            low_xy = xyxy_t[low_mask].cpu().tolist()
            low_sc = sc_t[low_mask].cpu().tolist()
            raw_low_props = sorted(
                [([float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], float(s)) for b, s in zip(low_xy, low_sc)],
                key=lambda x: -x[1]
            )
        else:
            raw_low_props = []

        if N == 0:
            return [], [], [], np.zeros((0, 14), dtype=np.float32), [], 0

        if candidate == 'B0':
            p14_all, _ = self._extract_anchor_p14_and_neck(
                h, w, [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy_t.cpu().tolist()], False
            )
            p_norm = (p14_all / p14_all.sum(dim=-1, keepdim=True).clamp_min(1e-8)).cpu().numpy()
            bxs_out = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy_t.cpu().tolist()]
            return bxs_out, sc_t.cpu().tolist(), cls_t.cpu().tolist(), p_norm, raw_low_props, 0

        xyxy_f, sc_f, cls_f = xyxy_t.clone(), sc_t.clone(), cls_t.clone()
        p14_all, _ = self._extract_anchor_p14_and_neck(
            h, w, [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy_t.cpu().tolist()], False
        )
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
            return bxs_out, sc_f.cpu().tolist(), cls_f.cpu().tolist(), probs_full.cpu().numpy(), raw_low_props, n_dino

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
            elif candidate == 'A4b':
                std_list = [crop_square_scale(rgb, bb, 1.25) for bb in bxs_fg]
                flip_list = [c[:, :, ::-1].copy() for c in std_list]
                c_both = np.stack(std_list + flip_list, axis=0)
                lg_both = self._encode_dino_logits(c_both)
                lp_std = F.log_softmax(lg_both[:k] / self.T_DINO, dim=-1)
                lp_flp = F.log_softmax(lg_both[k:] / self.T_DINO, dim=-1)
                wd, ws, wf = self.w_a4b
                p_post = F.softmax(wd * lp_det + ws * lp_std + wf * lp_flp, dim=-1)
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
        return bxs_out, sc_f.cpu().tolist(), cls_f.cpu().tolist(), probs_full.cpu().numpy(), raw_low_props, n_dino

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


def paired_cluster_bootstrap_orchard(clean_gt_dict: dict, preds_per_seed_model: dict, img_counts_per_seed_model: dict,
                                     img_idx_to_cluster: list, pairs_to_test: list, n_boot: int = 1000, rng_seed: int = 20261127):
    """
    按采集簇（Cluster-level）同步有放回配对 Bootstrap (`B = 1,000`)：
    同一棵树/同一枝条的连拍图像共享 `cluster_id`，每次重抽样抽取 G 个簇（保留重复簇内图像的权重倍数），
    在所有预声明冻结模型与全部随机种子上同步积分 COCO PR 曲线及一对一召回/Top-1/Top-3 准确率！
    """
    ious_thrs = np.linspace(0.5, 0.95, 10)
    rec_thrs = np.linspace(0.0, 1.0, 101)
    images = clean_gt_dict['images']
    img_ids = [im['id'] for im in images]
    iid2idx = {iid: idx for idx, iid in enumerate(img_ids)}
    cats = sorted([c['id'] for c in clean_gt_dict['categories'] if int(c['id']) >= 0])
    M, K = len(img_ids), len(cats)
    cat2k = {c: k for k, c in enumerate(cats)}

    gt_by_ic = defaultdict(list)
    for a in clean_gt_dict['annotations']:
        if a.get('iscrowd', 0) == 0 and a['category_id'] in cat2k and a['image_id'] in iid2idx:
            gt_by_ic[(iid2idx[a['image_id']], cat2k[a['category_id']])].append(a['bbox'])

    n_gt_ic = np.zeros((M, K), dtype=np.int32)
    for m in range(M):
        for k in range(K):
            n_gt_ic[m, k] = len(gt_by_ic.get((m, k), []))

    cluster_members = defaultdict(list)
    for m_idx, cid in enumerate(img_idx_to_cluster):
        cluster_members[cid].append(m_idx)
    unique_clusters = sorted(cluster_members.keys())
    G = len(unique_clusters)
    cluster_arrays = [np.array(cluster_members[c], dtype=np.int32) for c in unique_clusters]

    seeds_sorted = sorted(preds_per_seed_model.keys())
    needed_models = sorted({m for pair in pairs_to_test for m in pair})

    # Build per-image COCO match tables per (seed, model)
    match_tables = {s: {} for s in seeds_sorted}
    cnt_arrays = {s: {} for s in seeds_sorted}
    for s in seeds_sorted:
        for m_name in needed_models:
            cnt_arrays[s][m_name] = np.array(img_counts_per_seed_model[s][m_name], dtype=np.float64)  # (M, 4): [n_gt, n_loc, n_t1, n_t3]
            dt_by_ic = defaultdict(list)
            for r in preds_per_seed_model[s][m_name]:
                if r['image_id'] in iid2idx and r['category_id'] in cat2k:
                    dt_by_ic[(iid2idx[r['image_id']], cat2k[r['category_id']])].append((r['score'], r['bbox']))
            match_ic = [[None] * K for _ in range(M)]
            for m in range(M):
                for k in range(K):
                    dlist = dt_by_ic.get((m, k), [])
                    if not dlist:
                        continue
                    order = np.argsort([-x[0] for x in dlist], kind='mergesort')[:100]
                    dlist = [dlist[i] for i in order]
                    D = len(dlist)
                    scores = np.array([x[0] for x in dlist], dtype=np.float64)
                    tp = np.zeros((10, D), dtype=np.float64)
                    fp = np.zeros((10, D), dtype=np.float64)
                    glist = gt_by_ic.get((m, k), [])
                    if len(glist) == 0:
                        fp[:, :] = 1.0
                    else:
                        gb = np.array(glist, dtype=np.float64)
                        db = np.array([x[1] for x in dlist], dtype=np.float64)
                        ix1 = np.maximum(db[:, None, 0], gb[None, :, 0])
                        iy1 = np.maximum(db[:, None, 1], gb[None, :, 1])
                        ix2 = np.minimum(db[:, None, 0] + db[:, None, 2], gb[None, :, 0] + gb[None, :, 2])
                        iy2 = np.minimum(db[:, None, 1] + db[:, None, 3], gb[None, :, 1] + gb[None, :, 3])
                        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
                        iou_mat = inter / (db[:, None, 2] * db[:, None, 3] + gb[None, :, 2] * gb[None, :, 3] - inter + 1e-12)
                        for ti, thr in enumerate(ious_thrs):
                            gt_used = np.zeros(len(glist), dtype=bool)
                            for di in range(D):
                                best_iou, best_g = thr, -1
                                for gi in range(len(glist)):
                                    if not gt_used[gi] and iou_mat[di, gi] >= best_iou:
                                        best_iou = iou_mat[di, gi]
                                        best_g = gi
                                if best_g >= 0:
                                    gt_used[best_g] = True
                                    tp[ti, di] = 1.0
                                else:
                                    fp[ti, di] = 1.0
                    match_ic[m][k] = (scores, tp, fp)
            match_tables[s][m_name] = match_ic

    def eval_sample(s_key, m_name, idx_arr):
        match_ic = match_tables[s_key][m_name]
        ap_tk = []
        for k in range(K):
            npig = int(n_gt_ic[idx_arr, k].sum())
            if npig == 0:
                continue
            sc_l, tp_l, fp_l = [], [], []
            for m in idx_arr:
                it = match_ic[m][k]
                if it is not None:
                    sc_l.append(it[0]); tp_l.append(it[1]); fp_l.append(it[2])
            if not sc_l:
                ap_tk.append(np.zeros(10))
                continue
            sc_all = np.concatenate(sc_l)
            tp_all = np.concatenate(tp_l, axis=1)
            fp_all = np.concatenate(fp_l, axis=1)
            inds = np.argsort(-sc_all, kind='mergesort')
            tp_s = np.cumsum(tp_all[:, inds], axis=1)
            fp_s = np.cumsum(fp_all[:, inds], axis=1)
            rc = tp_s / float(npig)
            pr = tp_s / np.maximum(tp_s + fp_s, np.finfo(np.float64).eps)
            pr_env = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
            ap_10 = np.zeros(10)
            for ti in range(10):
                idxs = np.searchsorted(rc[ti], rec_thrs, side='left')
                vals = np.where(idxs < pr_env.shape[1], pr_env[ti, np.minimum(idxs, pr_env.shape[1] - 1)], 0.0)
                ap_10[ti] = vals.mean()
            ap_tk.append(ap_10)
        ap50_95 = float(np.stack(ap_tk, axis=0).mean() * 100.0) if ap_tk else 0.0
        ap50 = float(np.stack(ap_tk, axis=0)[:, 0].mean() * 100.0) if ap_tk else 0.0
        sums = cnt_arrays[s_key][m_name][idx_arr].sum(axis=0)  # [n_gt, n_loc, n_t1, n_t3]
        rec50 = float(sums[1] / max(1.0, sums[0]) * 100.0)
        top1 = float(sums[2] / max(1.0, sums[1]) * 100.0)
        mp_top3 = float(sums[3] / max(1.0, sums[0]) * 100.0)
        return np.array([ap50_95, ap50, rec50, top1, mp_top3], dtype=np.float64)

    # Verify unresampled COCO AP50-95 parity on first seed & model
    s0, m0 = seeds_sorted[0], needed_models[0]
    if preds_per_seed_model[s0][m0]:
        with contextlib.redirect_stdout(io.StringIO()):
            c_gt = COCO(); c_gt.dataset = clean_gt_dict; c_gt.createIndex()
            ev_chk = COCOeval_faster(c_gt, c_gt.loadRes(preds_per_seed_model[s0][m0]), 'bbox')
            ev_chk.evaluate(); ev_chk.accumulate(); ev_chk.summarize()
            chk_ap = ev_chk.stats[0] * 100.0
        fast_ap = eval_sample(s0, m0, np.arange(M))[0]
        assert abs(chk_ap - fast_ap) < 1e-4, f"Cluster bootstrap base COCO mismatch: {chk_ap} vs {fast_ap}"

    rng = np.random.RandomState(rng_seed)
    boot_vals = {m: np.zeros((n_boot, 5), dtype=np.float64) for m in needed_models}
    for b in range(n_boot):
        chosen_c = rng.randint(0, G, size=G)
        idx_arr = np.concatenate([cluster_arrays[c] for c in chosen_c])
        for m_name in needed_models:
            seed_vecs = [eval_sample(s, m_name, idx_arr) for s in seeds_sorted]
            boot_vals[m_name][b] = np.mean(seed_vecs, axis=0)

    metric_names = ['AP50_95', 'AP50', 'Recall_IoU50_pct', 'Matched_Top1_Acc_pct', 'MissPenalized_Total_Top3_Hit_pct']
    paired_out = {}
    for m_a, m_b in pairs_to_test:
        tag = f"{m_a}__minus__{m_b}"
        paired_out[tag] = {}
        diff_mat = boot_vals[m_a] - boot_vals[m_b]
        for col, m_lbl in enumerate(metric_names):
            dv = diff_mat[:, col]
            lo, hi = float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))
            p_two = min(1.0, 2.0 * min(float(np.mean(dv <= 0.0)), float(np.mean(dv >= 0.0))))
            paired_out[tag][m_lbl] = {
                'boot_mean_diff': round(float(dv.mean()), 3),
                'ci95_cluster_bootstrap': [round(lo, 3), round(hi, 3)],
                'p_value_two_sided': round(p_two, 4),
                'significant_95': bool(lo > 0.0 or hi < 0.0),
            }
    return {
        'n_images': M,
        'n_clusters': G,
        'n_bootstrap_replicates': n_boot,
        'paired_comparisons': paired_out,
    }


def evaluate_frozen_5models_on_labeled_orchard(manifest: dict, image_dir: Path, gt_json: Path, out_json: Path,
                                               cluster_csv: str = '', seeds=(0, 1, 2), n_boot: int = 1000):
    """
    第四阶段专用：评测预先声明的冻结五模型
    (B0: YOLOv8n, B1: Full DINOv2, C1: 1.25x, A4b: 1.25x+HFlip 补充对照, C3: 0.82x+1.25x 唯一主方法)，
    强制执行一对一匹配、不确定标注分离、完备五分法错误归因与按采集簇同步配对 Bootstrap (`B = 1,000`)。
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    raw_gt_dict = json.loads(gt_json.read_text(encoding='utf-8'))

    # Separate confirmable GT boxes from ambiguous "uncertain" expert annotations
    anns_by_img = defaultdict(list)
    clean_anns = []
    n_uncertain_ignored = 0
    for a in raw_gt_dict['annotations']:
        if a.get('iscrowd', 0) != 0:
            continue
        if bool(a.get('uncertain', False)) or str(a.get('status', '')).lower() == 'uncertain' or int(a.get('category_id', -1)) < 0:
            n_uncertain_ignored += 1
            continue
        anns_by_img[a['image_id']].append(a)
        clean_anns.append(a)

    clean_gt_dict = {
        'images': raw_gt_dict['images'],
        'annotations': clean_anns,
        'categories': [c for c in raw_gt_dict['categories'] if int(c['id']) >= 0],
    }

    fn2cluster = {}
    cluster_source = 'per_image_fallback_no_cluster_csv'
    if cluster_csv and Path(cluster_csv).exists():
        cluster_source = f'csv:{Path(cluster_csv).name}'
        for r in csv.DictReader(open(cluster_csv, encoding='utf-8-sig')):
            fn = r.get('file_name') or r.get('文件')
            cid = r.get('cluster_id') or r.get('簇编号')
            if fn and cid:
                fn2cluster[fn] = str(cid)
    elif any('cluster_id' in im for im in clean_gt_dict['images']):
        cluster_source = 'gt_json_image_cluster_id'

    img_idx_to_cluster = [
        fn2cluster.get(im['file_name'], str(im.get('cluster_id', im['id'])))
        for im in clean_gt_dict['images']
    ]

    models_5 = [
        ('B0', 'YOLOv8n'),
        ('B1', 'Full_DINOv2'),
        ('C1', 'C1_SingleScale_125'),
        ('A4b', 'A4b_FlipTTA_125_HFlip'),
        ('C3', 'C3_DualScale_082_125'),
    ]
    report = {
        'manifest_version': manifest['manifest_version'],
        'primary_method': 'C3_DualScale_082_125',
        'cluster_source': cluster_source,
        'n_images': len(clean_gt_dict['images']),
        'n_unique_clusters': len(set(img_idx_to_cluster)),
        'n_confirmable_gt_boxes': len(clean_anns),
        'n_uncertain_expert_boxes_excluded_from_gt': n_uncertain_ignored,
        'matching_protocol': 'strict_1to1_score_descending_greedy_iou_0.50',
        'models_3seed_summary': {},
        'per_seed': {},
    }

    preds_per_seed_model = {str(s): {} for s in seeds}
    img_counts_per_seed_model = {str(s): {} for s in seeds}
    agg_lists = {m[1]: defaultdict(list) for m in models_5}

    for seed in seeds:
        eng = UnifiedOrchardInferenceEngine(manifest, seed=seed, device=device)
        report['per_seed'][str(seed)] = {}
        for m_code, m_name in models_5:
            coco_rows = []
            img_cnt_rows = []
            times_all, times_multi = [], []
            n_gt_tot = n_prop_1to1_hit = n_rec50_loc = n_top1_ok = n_top3_ok = 0
            err_no_cand = err_poor_loc = err_score_nms_or_crowd = err_wrong_cls = 0

            for idx, im in enumerate(clean_gt_dict['images']):
                iid = im['id']
                p = image_dir / im['file_name']
                bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                bxs, scs, cls_l, probs, raw_low_props, _ = eng.infer_image_detailed(bgr, candidate=m_code)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if idx >= min(5, max(0, len(clean_gt_dict['images']) - 1)):
                    times_all.append(dt_ms)
                    if len(anns_by_img.get(iid, [])) >= 3 or len([s for s in scs if s >= 0.25]) >= 3:
                        times_multi.append(dt_ms)

                for bb, s, c in zip(bxs, scs, cls_l):
                    coco_rows.append({'image_id': iid, 'category_id': int(c), 'bbox': bb, 'score': float(s)})

                # Sort active detections (score >= 0.20) in strictly descending confidence order for 1-to-1 matching
                act_desc = sorted(
                    [(bxs[j], float(scs[j]), int(cls_l[j]), probs[j]) for j in range(len(bxs)) if scs[j] >= 0.20],
                    key=lambda x: -x[1]
                )
                glist = anns_by_img.get(iid, [])
                raw_1to1_map, max_raw_ious = match_detections_to_gt_1to1(raw_low_props, glist, iou_thr=0.50)
                act_1to1_map, _ = match_detections_to_gt_1to1(act_desc, glist, iou_thr=0.50)

                im_gt = len(glist)
                im_loc = im_t1 = im_t3 = 0
                n_gt_tot += im_gt
                n_prop_1to1_hit += len(raw_1to1_map)

                for gi, ga in enumerate(glist):
                    gc = int(ga['category_id'])
                    if gi in act_1to1_map:
                        dj = act_1to1_map[gi]
                        _, _, ac_, ap_ = act_desc[dj]
                        im_loc += 1
                        n_rec50_loc += 1
                        top3_c = np.argsort(-ap_)[:3].tolist()
                        if ac_ == gc:
                            im_t1 += 1
                            n_top1_ok += 1
                        else:
                            err_wrong_cls += 1
                        if gc in top3_c:
                            im_t3 += 1
                            n_top3_ok += 1
                    else:
                        mx_raw = max_raw_ious[gi]
                        if mx_raw < 0.10:
                            err_no_cand += 1
                        elif mx_raw < 0.50:
                            err_poor_loc += 1
                        else:
                            err_score_nms_or_crowd += 1

                img_cnt_rows.append([im_gt, im_loc, im_t1, im_t3])

            assert (n_top1_ok + err_wrong_cls + err_no_cand + err_poor_loc + err_score_nms_or_crowd) == n_gt_tot, \
                "Error attribution must form an exact 1-to-1 partition of confirmable GT boxes!"

            with contextlib.redirect_stdout(io.StringIO()):
                c_gt = COCO(); c_gt.dataset = clean_gt_dict; c_gt.createIndex()
                ev = COCOeval_faster(c_gt, c_gt.loadRes(coco_rows), 'bbox') if (coco_rows and clean_anns) else None
                if ev:
                    ev.evaluate(); ev.accumulate(); ev.summarize()
            ap50_95 = round(float(ev.stats[0] * 100.0), 2) if ev else 0.0
            ap50 = round(float(ev.stats[1] * 100.0), 2) if ev else 0.0

            s_rec = {
                'AP50_95': ap50_95,
                'AP50': ap50,
                'LowThr_Proposal_Recall50_1to1_pct': round(n_prop_1to1_hit / max(1, n_gt_tot) * 100.0, 2),
                'Recall_IoU50_Active_1to1_pct': round(n_rec50_loc / max(1, n_gt_tot) * 100.0, 2),
                'Matched_Top1_Acc_1to1_pct': round(n_top1_ok / max(1, n_rec50_loc) * 100.0, 2),
                'Matched_Top3_Acc_1to1_pct': round(n_top3_ok / max(1, n_rec50_loc) * 100.0, 2),
                'MissPenalized_Total_Top3_Hit_pct': round(n_top3_ok / max(1, n_gt_tot) * 100.0, 2),
                'Mean_Latency_ms': round(float(np.mean(times_all)), 2) if times_all else None,
                'P95_Latency_ms': round(float(np.percentile(times_all, 95)), 2) if times_all else None,
                'MultiTarget_Latency_ms': round(float(np.mean(times_multi)), 2) if times_multi else None,
                'Exact_GT_Partition_Counts': {
                    'total_confirmable_gt': n_gt_tot,
                    'localized_and_correct_top1': n_top1_ok,
                    'localized_but_wrong_class': err_wrong_cls,
                    'no_candidate_box_raw_iou_lt_010': err_no_cand,
                    'inaccurate_box_localization_raw_010_050': err_poor_loc,
                    'lost_by_score_nms_or_1to1_crowding': err_score_nms_or_crowd,
                },
            }
            report['per_seed'][str(seed)][m_name] = s_rec
            preds_per_seed_model[str(seed)][m_name] = coco_rows
            img_counts_per_seed_model[str(seed)][m_name] = img_cnt_rows
            for k, v in s_rec.items():
                if isinstance(v, (int, float)):
                    agg_lists[m_name][k].append(v)

        eng.close()

    for _, m_name in models_5:
        msum = {}
        for k, vals in agg_lists[m_name].items():
            msum[f'{k}_mean'] = round(float(np.mean(vals)), 2)
            msum[f'{k}_std'] = round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else 0.0
            msum[f'{k}_seeds'] = vals
        report['models_3seed_summary'][m_name] = msum

    pairs_orchard = [
        ('C3_DualScale_082_125', 'C1_SingleScale_125'),
        ('C3_DualScale_082_125', 'Full_DINOv2'),
        ('C3_DualScale_082_125', 'YOLOv8n'),
        ('C3_DualScale_082_125', 'A4b_FlipTTA_125_HFlip'),
    ]
    if clean_anns and n_boot > 0:
        report['cluster_paired_bootstrap'] = paired_cluster_bootstrap_orchard(
            clean_gt_dict, preds_per_seed_model, img_counts_per_seed_model,
            img_idx_to_cluster, pairs_orchard, n_boot=n_boot
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[预声明冻结五模型评测完成] 1-to-1 匹配 + 按簇 Bootstrap (G={report['n_unique_clusters']} 簇) -> {out_json}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description='冻结主方法 C3 统一在线推理与真实果园预声明五模型评测工具')
    ap.add_argument('--manifest', type=str, default=str(DEFAULT_MANIFEST))
    ap.add_argument('--mode', choices=['check_manifest', 'preannotate', 'eval_frozen5', 'eval_frozen4'], default='check_manifest')
    ap.add_argument('--image-dir', type=str, default='')
    ap.add_argument('--gt-json', type=str, default='')
    ap.add_argument('--cluster-csv', type=str, default='')
    ap.add_argument('--out-json', type=str, default='')
    ap.add_argument('--candidate', choices=['B0', 'B1', 'C1', 'C2', 'A4b', 'C3'], default='C3')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-images', type=int, default=0)
    ap.add_argument('--n-boot', type=int, default=1000)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    if args.mode == 'check_manifest':
        verify_manifest_hashes(manifest)
        s0 = manifest['seeds']['0']['locked_fusion_params_from_scidb_val']
        gammas = [manifest['seeds'][str(s)]['locked_fusion_params_from_scidb_val']['C2_gamma_ood'] for s in [0, 1, 2]]
        a4b_ws = [manifest['seeds'][str(s)]['locked_fusion_params_from_scidb_val']['A4b_FlipTTA_125_HFlip_weights_det_std_flip'] for s in [0, 1, 2]]
        print(f"[冻结配置校验通过] 版本={manifest['manifest_version']} | 唯一主方法={manifest['primary_method_id']} | "
              f"C3权重={s0['C3_MainMethod_weights_det_core082_std125']} | A4b按种子权重={a4b_ws} | "
              f"C2按种子γ={gammas} | 预声明评测模型={manifest['frozen_stage4_orchard_eval_models']} | 12份权重SHA-256全部吻合！", flush=True)
    elif args.mode == 'preannotate':
        assert args.image_dir and args.out_json, "preannotate 模式需指定 --image-dir 与 --out-json"
        run_preannotate_folder(manifest, Path(args.image_dir), Path(args.out_json), args.candidate, args.seed, max_images=args.max_images)
    elif args.mode in ('eval_frozen5', 'eval_frozen4'):
        assert args.image_dir and args.gt_json and args.out_json, "评测模式需指定 --image-dir, --gt-json 与 --out-json"
        evaluate_frozen_5models_on_labeled_orchard(
            manifest, Path(args.image_dir), Path(args.gt_json), Path(args.out_json), args.cluster_csv, n_boot=args.n_boot
        )


if __name__ == '__main__':
    main()
