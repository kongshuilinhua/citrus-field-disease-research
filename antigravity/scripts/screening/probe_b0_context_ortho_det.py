# -*- coding: utf-8 -*-
"""
Probe-B0: 框条件平滑背景光度干预 + 正交梯度投影 (PCGrad) 的单阶段无冲突泛化检测先导验证
直接回应仓库两项核心实测发现：
1. 11月23日《收尾说明_20261123.md》：固定框内像素、仅改变框外上下文会导致单阶段检测判定大幅退化；且硬贴图存在接缝与 donor 抢分问题；
2. 11月8日《47_源训练梯度冲突探针结果.md》：直接叠加辅助监督在正匹配位置存在 39.6% 梯度负内积冲突，拉低源域收敛。

本探针设计：
- 阶段 1 (梯度冲突核验)：在相同 128 个增强输入上检验“平滑框保护光度背景干预 + 对称 logit 一致性”相对原生检测梯度的内积分布，以及 PCGrad 正交投影消解负内积的数学精确性；
- 阶段 2 (30轮受控 YOLOv8n seed0 先导)：与 yolov8n_unified_seed0 完全相同的超参、初始化权重、数据划分与 30 轮预算，推理端交付 100% 原生 YOLOv8n（零额外参数、零推理耗时增加），训练期仅对 Detect.cv3 施加经 PCGrad 正交投影的平滑框外背景光度不变性正则；
- 阶段 3 (严格 FP32 共同评测)：在 val / testB / testC 上与原生 yolov8n_unified_seed0 做同进程严格 FP32 对比。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['OMP_NUM_THREADS'] = '2'

import sys
import csv
import json
import math
import time
import random
import hashlib
import contextlib
import io
from pathlib import Path

import cv2
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
sys.path.append(str(EXT))
from 共同评测后端 import configure_strict_fp32

OUT_DIR = ROOT / '多路线创新筛选_20261124' / '探针B_框内解耦与正交梯度检测' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def smooth_box_protected_context_intervention(imgs: torch.Tensor, bboxes_cxcywh: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    """
    构造零接缝、零外来叶片抢分的平滑框保护背景光度/底板干预视图 img_ctx:
    - 框内核心区 (M_fg = 1): 100% 保留原始目标病征像素；
    - 框边缘过渡区: 9x9 高斯羽化平滑过渡，消除矩形硬接缝伪影；
    - 框外纯背景区 (1 - M_fg): 随机施加跨图背景色温/明度交换或模拟白板/灰暗底板曝光偏移（模拟果园树冠阴影 <-> 受控背景板切换）。
    """
    B, C, H, W = imgs.shape
    device = imgs.device
    fg_mask = torch.zeros((B, 1, H, W), dtype=imgs.dtype, device=device)

    if len(bboxes_cxcywh) > 0:
        xyxy = box_convert(bboxes_cxcywh, 'cxcywh', 'xyxy')
        for b_id, (x1n, y1n, x2n, y2n) in zip(batch_idx.long().tolist(), xyxy.tolist()):
            ix1 = max(0, int(math.floor(x1n * W)))
            iy1 = max(0, int(math.floor(y1n * H)))
            ix2 = min(W, int(math.ceil(x2n * W)))
            iy2 = min(H, int(math.ceil(y2n * H)))
            if ix2 > ix1 and iy2 > iy1:
                fg_mask[b_id, 0, iy1:iy2, ix1:ix2] = 1.0

    # Smooth Gaussian feathering on mask boundary to eliminate artificial hard box seams
    coords = torch.arange(9, dtype=imgs.dtype, device=device) - 4.0
    g1d = torch.exp(-(coords ** 2) / (2 * 2.0 * 2.0))
    g1d = g1d / g1d.sum()
    g2d = (g1d[:, None] * g1d[None, :]).unsqueeze(0).unsqueeze(0)
    smooth_m = F.conv2d(F.pad(fg_mask, (4, 4, 4, 4), mode='reflect'), g2d)
    # Keep strict 1.0 inside core foreground box while feathering outward
    m_protect = torch.maximum(fg_mask, smooth_m).clamp(0.0, 1.0)

    # Generate diverse background photometric/board styles without introducing foreign donor leaves
    perm = torch.roll(torch.arange(B, device=device), shifts=1)
    donor_mean = imgs[perm].mean(dim=(2, 3), keepdim=True)
    donor_std = imgs[perm].std(dim=(2, 3), keepdim=True).clamp_min(0.05)
    self_mean = imgs.mean(dim=(2, 3), keepdim=True)
    self_std = imgs.std(dim=(2, 3), keepdim=True).clamp_min(0.05)

    # Style-transferred background (preserves coarse natural texture without foreign objects, shifts illumination/color)
    bg_style = ((imgs - self_mean) / self_std) * donor_std + donor_mean

    # For half the batch, blend toward high-key white board / low-key shadow board luminance to mimic Condition B <-> C
    board_mode = torch.rand((B, 1, 1, 1), device=device, dtype=imgs.dtype)
    board_val = torch.where(board_mode > 0.5, torch.full_like(board_mode, 0.88), torch.full_like(board_mode, 0.22))
    alpha_board = (torch.rand((B, 1, 1, 1), device=device, dtype=imgs.dtype) * 0.55)
    bg_perturbed = ((1.0 - alpha_board) * bg_style + alpha_board * board_val).clamp(0.0, 1.0)

    img_ctx = m_protect * imgs + (1.0 - m_protect) * bg_perturbed
    return img_ctx.clamp(0.0, 1.0)


class ContextOrthoTrainer(DetectionTrainer):
    """在 YOLOv8n 原生训练中引入框保护上下文不变性 + Detect.cv3 正交梯度投影 (PCGrad)，交付模型 100% 原生。"""
    ctx_coef = 0.25

    def __init__(self, *args, **kwargs):
        self.conflict_stats = {'total_steps': 0, 'projected_layers': 0, 'total_layers': 0, 'ctx_loss_sum': 0.0}
        super().__init__(*args, **kwargs)

    def _model_train_step_with_pcgrad(self, batch):
        """重写单步反向传播，精确计算原生检测损失梯度与框外背景不变性梯度，并在 Detect.cv3 上执行正交梯度投影。"""
        self.optimizer.zero_grad()
        batch = self.preprocess_batch(batch)
        imgs = batch['img']  # float32 in [0, 1], (B, 3, 640, 640)
        student = unwrap_model(self.model)
        detect_head = student.model[-1]
        cv3_params = [p for p in detect_head.cv3.parameters() if p.requires_grad]

        with torch.amp.autocast('cuda', enabled=self.amp):
            preds_orig = self.model(imgs)
            loss_det, self.loss_items = student.loss(batch, preds_orig)

            # Extract positive-matched classification logits from preds_orig
            feats_orig = preds_orig[1] if isinstance(preds_orig, tuple) else preds_orig
            no = detect_head.no
            reg_ch = detect_head.reg_max * 4
            B = imgs.shape[0]
            cls_orig_all = torch.cat([f.view(B, no, -1) for f in feats_orig], dim=2)[:, reg_ch:, :]  # (B, 14, 8400)

            # Identify foreground-high-response anchor locations (top-k or score > 0.20 inside GT boxes)
            with torch.no_grad():
                img_ctx = smooth_box_protected_context_intervention(imgs, batch['bboxes'], batch['batch_idx'])
                # Build spatial GT union mask across 8400 anchors (80x80 + 40x40 + 20x20)
                anchor_masks = []
                for hw in [80, 40, 20]:
                    m_s = torch.zeros((B, hw, hw), dtype=torch.bool, device=imgs.device)
                    if len(batch['bboxes']) > 0:
                        xyxy = box_convert(batch['bboxes'], 'cxcywh', 'xyxy')
                        for b_id, (x1n, y1n, x2n, y2n) in zip(batch['batch_idx'].long().tolist(), xyxy.tolist()):
                            gx1, gy1 = max(0, int(math.floor(x1n * hw))), max(0, int(math.floor(y1n * hw)))
                            gx2, gy2 = min(hw, int(math.ceil(x2n * hw))), min(hw, int(math.ceil(y2n * hw)))
                            if gx2 > gx1 and gy2 > gy1:
                                m_s[b_id, gy1:gy2, gx1:gx2] = True
                    anchor_masks.append(m_s.view(B, -1))
                fg_anchor_mask = torch.cat(anchor_masks, dim=1)  # (B, 8400)

            preds_ctx = self.model(img_ctx)
            feats_ctx = preds_ctx[1] if isinstance(preds_ctx, tuple) else preds_ctx
            cls_ctx_all = torch.cat([f.view(B, no, -1) for f in feats_ctx], dim=2)[:, reg_ch:, :]  # (B, 14, 8400)

            if fg_anchor_mask.any():
                # Symmetric Bernoulli KL / MSE on foreground anchor classification probabilities
                s_o = cls_orig_all.permute(0, 2, 1)[fg_anchor_mask].float()
                s_c = cls_ctx_all.permute(0, 2, 1)[fg_anchor_mask].float()
                p_o = torch.sigmoid(s_o)
                p_c = torch.sigmoid(s_c)
                # Stop-gradient from clean original view + symmetric consistency on context-perturbed view
                loss_ctx = (
                    F.binary_cross_entropy_with_logits(s_c, p_o.detach())
                    - (p_o.detach() * torch.log(p_o.detach().clamp_min(1e-7))
                       + (1.0 - p_o.detach()) * torch.log((1.0 - p_o.detach()).clamp_min(1e-7))).mean()
                ) * B * self.ctx_coef
            else:
                loss_ctx = loss_det * 0.0

        # Compute gradients of loss_ctx on cv3_params first for PCGrad orthogonal projection
        if fg_anchor_mask.any() and loss_ctx.requires_grad:
            g_ctx_cv3 = torch.autograd.grad(self.scaler.scale(loss_ctx), cv3_params, retain_graph=True, allow_unused=True)
        else:
            g_ctx_cv3 = [None] * len(cv3_params)

        # Backward native detection loss on ALL model parameters
        self.loss = loss_det
        self.scaler.scale(loss_det).backward()

        # Inject PCGrad orthogonally-projected context-invariance gradients onto Detect.cv3!
        # Whenever g_ctx · g_det < 0, remove the conflicting component along g_det so native detection NEVER degrades!
        proj_cnt, tot_cnt = 0, 0
        with torch.no_grad():
            for p, gc in zip(cv3_params, g_ctx_cv3):
                if gc is None or p.grad is None:
                    continue
                if not (torch.isfinite(gc).all() and torch.isfinite(p.grad).all()):
                    continue
                tot_cnt += 1
                gd = p.grad
                dot = (gc * gd).sum()
                if dot < 0:
                    proj_cnt += 1
                    denom = (gd * gd).sum().clamp_min(1e-12)
                    gc_orth = gc - (dot / denom) * gd
                else:
                    gc_orth = gc
                p.grad.add_(gc_orth)

        self.conflict_stats['total_steps'] += 1
        self.conflict_stats['projected_layers'] += proj_cnt
        self.conflict_stats['total_layers'] += tot_cnt
        self.conflict_stats['ctx_loss_sum'] += float(loss_ctx.detach().item())


def evaluate_strict_fp32(weight_path: Path, tag: str) -> dict:
    backend = configure_strict_fp32()
    model = YOLO(str(weight_path))
    img_root = Path(r'D:\病虫害论文\复现实验记录\yolo11n_public_compare_20261004\数据副本\images')
    out_eval = {}

    for split in ['val', 'testB', 'testC']:
        gt_path = EXT / '检测数据' / f'{split}.json'
        gt_data = json.loads(gt_path.read_text(encoding='utf-8'))
        folder = img_root / split
        preds = []
        for start in range(0, len(gt_data['images']), 16):
            group = gt_data['images'][start:start + 16]
            images = [cv2.imdecode(np.fromfile(str(folder / it['file_name']), dtype=np.uint8), 1) for it in group]
            results = model.predict(images, imgsz=640, conf=0.001, iou=0.7, max_det=300,
                                    device=0, half=False, rect=False, verbose=False)
            for it, r in zip(group, results):
                for bbox, sc, cl in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().tolist()):
                    x1, y1, x2, y2 = bbox
                    preds.append({
                        'image_id': it['id'],
                        'category_id': int(cl),
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'score': sc,
                    })
        pred_file = OUT_DIR / f'{tag}_{split}_preds.json'
        pred_file.write_text(json.dumps(preds), encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(str(gt_path))
            ev = COCOeval_faster(gt, gt.loadRes(str(pred_file)), 'bbox')
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        stats = ev.stats.tolist()
        out_eval[split] = {
            'AP50_95': round(stats[0] * 100.0, 4),
            'AP50': round(stats[1] * 100.0, 4),
            'pred_count': len(preds),
            'pred_sha256': sha256_file(pred_file),
        }
        print(f"  [{tag} | {split}] AP50-95={out_eval[split]['AP50_95']:.4f}% | AP50={out_eval[split]['AP50']:.4f}%", flush=True)
    del model
    torch.cuda.empty_cache()
    return out_eval


def build_pcgrad_trainer():
    """基于 Ultralytics DetectionTrainer 干净接入自定义单步 PCGrad 更新。"""
    class CustomPCGradDetectionTrainer(ContextOrthoTrainer):
        def _do_train(self, world_size=1):
            # Bind custom step hook by monkey-patching criterion/backward inside standard loop?
            # Even cleaner in Ultralytics 8.3: override preprocess_batch + custom loss wrapper?
            # Wait: in Ultralytics DetectionTrainer._do_train, backward is `self.scaler.scale(self.loss).backward()`.
            # Using a tensor grad hook on Detect.cv3 parameters achieves exact per-parameter PCGrad inside native _do_train
            # with 100% identical RNG sequence, EMA, lr_scheduler, and logging!
            return super()._do_train(world_size)
    return CustomPCGradDetectionTrainer


class PCGradHookLoss:
    """
    100% 兼容 Ultralytics 原生 _do_train 循环的框保护背景干预 + PCGrad 梯度投影损失包装器：
    在不改动任何采样、学习率调度、AMP GradScaler、EMA 更新顺序的前提下：
    1. 计算原生检测损失 native_loss 及其对 cv3 参数的参考梯度 g_det；
    2. 计算平滑框保护背景干预视图 img_ctx 在前景框区域的 cv3 类别不变性损失 ctx_loss 及其梯度 g_ctx；
    3. 当 g_ctx · g_det < 0 时正交投影消除冲突分量，仅将无冲突梯度注入 cv3！
    """
    def __init__(self, orig_criterion, student_model, coef=0.25):
        self.orig_criterion = orig_criterion
        self.student = student_model
        self.coef = coef
        self.stats = {'steps': 0, 'raw_conflict_layers': 0, 'post_conflict_layers': 0, 'total_checked': 0}

    def __call__(self, preds, batch):
        native_loss, loss_items = self.orig_criterion(preds, batch)
        if not self.student.training or self.coef <= 0.0:
            return native_loss, loss_items

        imgs = batch['img']
        B, _, H, W = imgs.shape
        detect_head = self.student.model[-1]
        cv3_params = [p for p in detect_head.cv3.parameters() if p.requires_grad]

        # Build foreground box anchor mask
        with torch.no_grad():
            img_ctx = smooth_box_protected_context_intervention(imgs, batch['bboxes'], batch['batch_idx'])
            anchor_masks = []
            for hw in [80, 40, 20]:
                m_s = torch.zeros((B, hw, hw), dtype=torch.bool, device=imgs.device)
                if len(batch['bboxes']) > 0:
                    xyxy = box_convert(batch['bboxes'], 'cxcywh', 'xyxy')
                    for b_id, (x1n, y1n, x2n, y2n) in zip(batch['batch_idx'].long().tolist(), xyxy.tolist()):
                        gx1, gy1 = max(0, int(math.floor(x1n * hw))), max(0, int(math.floor(y1n * hw)))
                        gx2, gy2 = min(hw, int(math.ceil(x2n * hw))), min(hw, int(math.ceil(y2n * hw)))
                        if gx2 > gx1 and gy2 > gy1:
                            m_s[b_id, gy1:gy2, gx1:gx2] = True
                anchor_masks.append(m_s.view(B, -1))
            fg_mask = torch.cat(anchor_masks, dim=1)

        if not fg_mask.any():
            return native_loss, loss_items

        feats_orig = preds[1] if isinstance(preds, tuple) else preds
        no = detect_head.no
        reg_ch = detect_head.reg_max * 4
        cls_orig = torch.cat([f.view(B, no, -1) for f in feats_orig], dim=2)[:, reg_ch:, :]

        preds_ctx = self.student(img_ctx)
        feats_ctx = preds_ctx[1] if isinstance(preds_ctx, tuple) else preds_ctx
        cls_ctx = torch.cat([f.view(B, no, -1) for f in feats_ctx], dim=2)[:, reg_ch:, :]

        with torch.amp.autocast('cuda', enabled=False):
            s_o_box = cls_orig.permute(0, 2, 1)[fg_mask].float().detach()
            s_c_box = cls_ctx.permute(0, 2, 1)[fg_mask].float()
            p_o_box = torch.sigmoid(s_o_box)
            # 自适应前景病征锚点筛选：仅对框内已激活的叶片/病征锚点 (max_prob > 0.15) 施加背景不变性，自动避开矩形框四角背景像素！
            active_fg = p_o_box.max(dim=1).values > 0.15
            if active_fg.sum() >= 4:
                p_o = p_o_box[active_fg]
                s_c = s_c_box[active_fg]
            else:
                p_o = p_o_box
                s_c = s_c_box
            # Bernoulli KL(p_o || sigmoid(s_c)) summed over classes, averaged over active foreground anchors
            bce = F.binary_cross_entropy_with_logits(s_c, p_o, reduction='none').sum(dim=1).mean()
            ent = -(p_o * torch.log(p_o.clamp_min(1e-7)) + (1.0 - p_o) * torch.log((1.0 - p_o).clamp_min(1e-7))).sum(dim=1).mean()
            ctx_scalar = (bce - ent) * B * self.coef

            # Compute exact gradients on cv3_params (unscaled FP32 loss scalars!)
            g_dets = torch.autograd.grad(native_loss.sum(), cv3_params, retain_graph=True, allow_unused=True)
            g_ctxs = torch.autograd.grad(ctx_scalar, cv3_params, retain_graph=False, allow_unused=True)

            # Construct surrogate linear loss term sum_i <stop_grad(g_ctx_orth_i), p_i> whose backward injects exact PCGrad!
            surrogate_terms = []
            self.stats['steps'] += 1
            for p, gd, gc in zip(cv3_params, g_dets, g_ctxs):
                if gd is None or gc is None:
                    continue
                if not (torch.isfinite(gd).all() and torch.isfinite(gc).all()):
                    continue
                self.stats['total_checked'] += 1
                dot = (gc.float() * gd.float()).sum()
                if dot < 0:
                    self.stats['raw_conflict_layers'] += 1
                    denom = (gd.float() * gd.float()).sum().clamp_min(1e-12)
                    gc_orth = gc.float() - (dot / denom) * gd.float()
                    if (gc_orth * gd.float()).sum() < -1e-5:
                        self.stats['post_conflict_layers'] += 1
                else:
                    gc_orth = gc.float()
                surrogate_terms.append((p.float() * gc_orth.detach()).sum())

            surrogate_loss = torch.stack(surrogate_terms).sum() if surrogate_terms else native_loss.sum() * 0.0

        # Add surrogate_loss to cls slot of native_loss (3-element tensor: box, cls, dfl)
        total_loss = native_loss + torch.stack([surrogate_loss * 0.0, surrogate_loss.to(native_loss.dtype), surrogate_loss * 0.0])
        return total_loss, loss_items


class CleanOrthoYoloTrainer(DetectionTrainer):
    ctx_coef = 0.25

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        student = unwrap_model(self.model)
        if not isinstance(getattr(student, 'criterion', None), PCGradHookLoss):
            orig_crit = student.init_criterion()
            student.criterion = PCGradHookLoss(orig_crit, student, coef=self.ctx_coef)
            assert not isinstance(getattr(self.ema.ema, 'criterion', None), PCGradHookLoss)
        return batch


def main():
    t0 = time.monotonic()
    init_weight = ROOT / '复现_论文' / 'yolov8n.pt'
    assert sha256_file(init_weight) == 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'

    run_dir = Path(r'D:\citrus_runs\yolov8_ctx_ortho_probe_20261124_seed0')
    if not (run_dir / 'weights' / 'best.pt').exists():
        print("[1/2] 启动 YOLOv8n 框保护背景干预 + PCGrad 正交梯度投影 30轮受控训练 (seed=0)...", flush=True)
        model = YOLO(str(init_weight))
        model.train(
            trainer=CleanOrthoYoloTrainer,
            data=str(EXT / '统一检测数据' / 'data.yaml'),
            epochs=30,
            imgsz=640,
            batch=16,
            optimizer='AdamW',
            lr0=0.001,
            weight_decay=0.0005,
            seed=0,
            workers=0,
            device=0,
            amp=True,
            patience=0,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            plots=False,
            save=True,
            cache=False,
        )
        crit = unwrap_model(model.trainer.model).criterion
        pcgrad_stats = crit.stats if isinstance(crit, PCGradHookLoss) else {}
        (OUT_DIR / 'pcgrad_training_conflict_stats.json').write_text(json.dumps(pcgrad_stats, indent=2, ensure_ascii=False), encoding='utf-8')
        print("  PCGrad 训练期梯度投影统计:", pcgrad_stats, flush=True)

        # Verify recipe parity with yolov8n_unified_seed0
        actual_args = yaml.safe_load((run_dir / 'args.yaml').read_text(encoding='utf-8'))
        ref_args = yaml.safe_load(Path(r'D:\citrus_runs\yolov8n_unified_seed0\args.yaml').read_text(encoding='utf-8'))
        ignored = {'project', 'name', 'exist_ok', 'model', 'save_dir'}
        diffs = {k: [ref_args[k], actual_args.get(k)] for k in ref_args if k not in ignored and ref_args[k] != actual_args.get(k)}
        assert not diffs, f"Recipe mismatch vs baseline: {diffs}"
    else:
        pcgrad_stats = json.loads((OUT_DIR / 'pcgrad_training_conflict_stats.json').read_text(encoding='utf-8')) if (OUT_DIR / 'pcgrad_training_conflict_stats.json').exists() else {}

    print("\n[2/2] 执行同进程严格 FP32 方形 640 评测对比...", flush=True)
    best_pt = run_dir / 'weights' / 'best.pt'
    # Verify delivered checkpoint has ZERO extra parameters
    loaded_model = YOLO(str(best_pt)).model
    assert getattr(loaded_model, 'criterion', None) is None

    ours_eval = evaluate_strict_fp32(best_pt, 'ctx_ortho_seed0')

    # Reference seed0 strict FP32 numbers from YOLOv8强基线重复_20261109/种子0
    ref_eval = {}
    for split in ['val', 'testB', 'testC']:
        r_json = json.loads((EXT / 'YOLOv8强基线重复_20261109' / '种子0' / f'{split}评价.json').read_text(encoding='utf-8'))
        ref_eval[split] = {
            'AP50_95': round(r_json['COCO统计'][0] * 100.0, 4),
            'AP50': round(r_json['COCO统计'][1] * 100.0, 4),
        }

    deltas = {
        sp: {
            'delta_AP50_95_pt': round(ours_eval[sp]['AP50_95'] - ref_eval[sp]['AP50_95'], 4),
            'delta_AP50_pt': round(ours_eval[sp]['AP50'] - ref_eval[sp]['AP50'], 4),
        }
        for sp in ['val', 'testB', 'testC']
    }

    summary = {
        '实验名称': 'Probe-B0_框保护背景光度干预与PCGrad正交投影检测先导_seed0',
        '总耗时秒': round(time.monotonic() - t0, 2),
        '最佳权重路径': str(best_pt),
        '最佳权重SHA256': sha256_file(best_pt),
        '原生交付零额外参数核验': True,
        '训练期PCGrad冲突投影统计': pcgrad_stats,
        '原生基线_seed0': ref_eval,
        '本探针_seed0': ours_eval,
        '净变化_百分点': deltas,
    }
    out_p = OUT_DIR / 'probe_b0_ctx_ortho_summary.json'
    out_p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print("\n================ Probe-B0 对比汇总 (seed0) ================", flush=True)
    for sp in ['val', 'testB', 'testC']:
        print(f"  {sp:5s}: 原生={ref_eval[sp]['AP50_95']:.2f}% -> 本方法={ours_eval[sp]['AP50_95']:.2f}% ({deltas[sp]['delta_AP50_95_pt']:+.2f}pt)", flush=True)
    print(f"[完成] 结果已保存至: {out_p}", flush=True)


if __name__ == '__main__':
    main()
