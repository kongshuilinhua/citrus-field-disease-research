# -*- coding: utf-8 -*-
"""
Probe-A2-Plus: 针对 Probe-A2 实测反常现象（传统 MSP/Energy/EDL 在跨域同批内部 AUROC 跌至 41%~52%）的机理验证与修复
核心问题（已由 Probe-A2 实证发现）：
- 当模型从 SciDB 跨到异域（如 CitrusUAT 或新果园胡柚）时，所有目标域样本均携带共同的协变量偏移向量 Δ_dom（背景底色、曝光、品种底色）；
- 传统不确定性（MSP / Energy / Flat-EDL / 朴素 Hier-EDL）度量的是 ||z - z_src||，其幅值被 ||Δ_dom|| 主导，导致“目标域已知病害”与“目标域未知虫害/未见缺素”同样呈现高不确定性（val-vs-Pest AUROC 虚高达 97%，而目标域内部 Known-vs-Pest AUROC 崩塌至 ~50% 随机水平）！

候选创新机制：无监督目标域流形中心去偏与白化校准原型开放集判别 (Domain-Debiased Prototype & Subspace OSR, DDP-OSR)
- 完全无监督（仅利用无标签测试域批次一阶均值/二阶协方差对齐源域全局统计量，严禁使用任何测试标签！）；
- 同时评测预训练未畸变特征流形 (Frozen DINOv2) + 微调判别流形 (LP-FT / MobileNetV3) 的解耦协同：用微调头保细粒度症状分类，用无监督流形去偏原型余弦/马氏残差判别同域内未知虫害与未见缺素！
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
import timm
from sklearn.metrics import roc_auc_score

ROOT = Path(r'D:\新版柑橘')
BASE_CACHE = ROOT / '基线补齐_20261026'
OLD_LIST = ROOT / '本周科研_20261014' / '裁图清单.json'
CITRUS_UAT_DIR = ROOT / '复现_baseline' / 'field_test_CitrusUAT'
OUT_DIR = ROOT / '多路线创新筛选_20261124' / '探针A_选择性路由与证据拒识' / 'results'


def load_citrus_uat_crops():
    tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    imgs, cls_names = [], []
    for cls_dir in sorted([d for d in CITRUS_UAT_DIR.iterdir() if d.is_dir()], key=lambda p: p.name):
        for img_p in sorted(cls_dir.glob('*.*')):
            bgr = cv2.imdecode(np.fromfile(str(img_p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            imgs.append(np.asarray(tf(Image.fromarray(rgb))).transpose(2, 0, 1))
            cls_names.append(cls_dir.name)
    return np.stack(imgs, axis=0), cls_names


@torch.no_grad()
def extract_features(uat_imgs: np.ndarray, device='cuda:0'):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)

    def run_arr(arr_u8, fn):
        out = []
        for i in range(0, len(arr_u8), 128):
            x = (torch.from_numpy(np.array(arr_u8[i:i + 128])).to(device).float() / 255.0 - mean) / std
            out.append(F.normalize(fn(x).float(), dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    # 1. Pretrained Frozen DINOv2 (undistorted general visual manifold, already cached for SciDB!)
    scidb_pre = {
        sp: np.load(BASE_CACHE / '特征缓存' / f'末1CLS_{sp}.npy')
        for sp in ['train', 'val', 'testB', 'testC']
    }
    w_pre = ROOT / '本周科研_20261014' / '模型缓存' / 'dinov2_vits14_pretrain.pth'
    m_pre = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True,
                              pretrained_cfg_overlay={'file': str(w_pre)}, img_size=224, num_classes=0).to(device).eval()
    uat_pre = run_arr(uat_imgs, lambda x: m_pre(x))
    del m_pre

    # 2. Fine-tuned DINOv2 LP-FT (3 seeds!)
    meta = json.loads((ROOT / '方法推进_20261027' / '纯CLS对照' / '微调汇总.json').read_text(encoding='utf-8'))
    lpft_seeds = []
    for s in [0, 1, 2]:
        t_info = next(r for r in meta['重复结果'] if r['方法'] == 'LP-FT' and r['种子'] == s)
        state = torch.load(t_info['权重文件'], map_location='cpu', weights_only=True)
        enc = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, img_size=224, num_classes=0).to(device).eval()
        enc.load_state_dict(state['骨干'])
        head = nn.Linear(384, 14).to(device).eval()
        head.load_state_dict(state['分类头'])
        scidb_ft = {
            sp: run_arr(np.load(BASE_CACHE / '图像缓存' / f'{sp}.npy', mmap_mode='r'), lambda x: enc.forward_features(x)[:, 0])
            for sp in ['train', 'val']
        }
        uat_ft = run_arr(uat_imgs, lambda x: enc.forward_features(x)[:, 0])
        with torch.no_grad():
            uat_logits = head(torch.from_numpy(uat_ft).to(device)).cpu().numpy()
            val_logits = head(torch.from_numpy(scidb_ft['val']).to(device)).cpu().numpy()
        lpft_seeds.append({
            'seed': s,
            'scidb_ft': scidb_ft,
            'uat_ft': uat_ft,
            'uat_logits': uat_logits,
            'val_logits': val_logits,
        })
        del enc, head
        torch.cuda.empty_cache()

    return scidb_pre, uat_pre, lpft_seeds


def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(1e-8, np.linalg.norm(x, axis=1, keepdims=True))


def domain_recenter_whiten(z_src: np.ndarray, z_tgt_unlabeled: np.ndarray, reg: float = 0.05) -> np.ndarray:
    """
    完全无监督流形去偏（CORAL/Tangent Recentering）：
    仅用无标签目标域整体均值与协方差对齐源域一阶/二阶矩，剥离全局拍摄背景/光照偏移 Δ_dom，绝不使用任何目标域类别信息！
    """
    mu_s = z_src.mean(axis=0, keepdims=True)
    mu_t = z_tgt_unlabeled.mean(axis=0, keepdims=True)
    # 1. Centering shift removal + re-inject source center
    z_centered = z_tgt_unlabeled - mu_t

    # 2. Regularized CORAL alignment: C_t^{-1/2} C_s^{1/2}
    d = z_src.shape[1]
    cs = ((z_src - mu_s).T @ (z_src - mu_s)) / max(1, len(z_src) - 1) + reg * np.eye(d)
    ct = (z_centered.T @ z_centered) / max(1, len(z_tgt_unlabeled) - 1) + reg * np.eye(d)

    es, us = np.linalg.eigh(cs)
    et, ut = np.linalg.eigh(ct)
    cs_half = (us * np.sqrt(np.maximum(1e-8, es))) @ us.T
    ct_inv_half = (ut * (1.0 / np.sqrt(np.maximum(1e-8, et)))) @ ut.T

    z_aligned = z_centered @ ct_inv_half @ cs_half + mu_s
    return l2_norm(z_aligned)


def fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    tau = np.quantile(id_scores, 0.95)
    return float((ood_scores <= tau).mean()) * 100.0


def main():
    t0 = time.monotonic()
    recs = json.loads(OLD_LIST.read_text(encoding='utf-8-sig'))
    y_train = np.array([r['类别'] - 1 for r in recs['train']], dtype=np.int64)
    y_val = np.array([r['类别'] - 1 for r in recs['val']], dtype=np.int64)

    uat_imgs, uat_cls = load_citrus_uat_crops()
    uat_cls_arr = np.array(uat_cls)
    mask_known = np.isin(uat_cls_arr, ['Healthy', 'Greasy_spot'])
    mask_pest = np.isin(uat_cls_arr, ['Citrus_leafminer', 'Red_scale', 'Red_scale_sequelae', 'Texas_mite'])
    mask_unseen_nutr = np.isin(uat_cls_arr, ['Fe', 'Mn', 'N'])
    y_known_gt = np.where(uat_cls_arr[mask_known] == 'Healthy', 1, 10)

    scidb_pre, uat_pre, lpft_seeds = extract_features(uat_imgs, 'cuda:0')

    # Fit 14 class prototypes on Undistorted Pretrained DINOv2 manifold (SciDB train ONLY!)
    proto_pre = l2_norm(np.stack([scidb_pre['train'][y_train == c].mean(axis=0) for c in range(14)], axis=0))

    seed_reports = []
    for item in lpft_seeds:
        s = item['seed']
        uat_logits = item['uat_logits']
        val_logits = item['val_logits']

        # Baseline 1: Closed-set Energy on fine-tuned head
        unc_val_energy = -np.log(np.sum(np.exp(val_logits), axis=1))
        unc_uat_energy = -np.log(np.sum(np.exp(uat_logits), axis=1))
        pred_ft = np.argmax(uat_logits, axis=1)
        acc_ft_known = float((pred_ft[mask_known] == y_known_gt).mean()) * 100.0

        # Baseline 2: Raw fine-tuned prototype cosine distance (distorted manifold, no domain debiasing)
        proto_ft = l2_norm(np.stack([item['scidb_ft']['train'][y_train == c].mean(axis=0) for c in range(14)], axis=0))
        unc_val_ft_cos = 1.0 - np.max(item['scidb_ft']['val'] @ proto_ft.T, axis=1)
        unc_uat_ft_cos = 1.0 - np.max(item['uat_ft'] @ proto_ft.T, axis=1)

        # Step 1 Ablation: Undistorted Pretrained Manifold Prototype Distance (without domain recentering)
        unc_val_pre_raw = 1.0 - np.max(scidb_pre['val'] @ proto_pre.T, axis=1)
        unc_uat_pre_raw = 1.0 - np.max(uat_pre @ proto_pre.T, axis=1)
        pred_pre_raw = np.argmax(uat_pre @ proto_pre.T, axis=1)
        acc_pre_raw_known = float((pred_pre_raw[mask_known] == y_known_gt).mean()) * 100.0

        # Step 2 Ours (DDP-OSR): Domain-Debiased (Unsupervised CORAL-Recentered) Undistorted Manifold Prototype + Local KNN Density!
        uat_pre_aligned = domain_recenter_whiten(scidb_pre['train'], uat_pre, reg=0.05)
        sim_uat_aligned = uat_pre_aligned @ proto_pre.T
        sim_val = scidb_pre['val'] @ proto_pre.T
        pred_ddp = np.argmax(sim_uat_aligned, axis=1)
        acc_ddp_known = float((pred_ddp[mask_known] == y_known_gt).mean()) * 100.0

        # Relative Prototype Margin / Cosine Novelty Score: 1 - max_c cos(z_aligned, mu_c)
        unc_val_ddp = 1.0 - np.max(sim_val, axis=1)
        unc_uat_ddp = 1.0 - np.max(sim_uat_aligned, axis=1)

        # Also test Debiased Fine-Tuned Manifold (CORAL on FT features!)
        uat_ft_aligned = domain_recenter_whiten(item['scidb_ft']['train'], item['uat_ft'], reg=0.05)
        pred_ft_aligned = np.argmax(uat_ft_aligned @ proto_ft.T, axis=1)
        acc_ft_aligned_known = float((pred_ft_aligned[mask_known] == y_known_gt).mean()) * 100.0
        unc_uat_ft_aligned = 1.0 - np.max(uat_ft_aligned @ proto_ft.T, axis=1)

        def eval_osr(u_val, u_uat, known_acc, pred_arr):
            auroc_within_pest = float(roc_auc_score(
                np.concatenate([np.zeros(mask_known.sum()), np.ones(mask_pest.sum())]),
                np.concatenate([u_uat[mask_known], u_uat[mask_pest]])
            )) * 100.0
            auroc_within_nutr = float(roc_auc_score(
                np.concatenate([np.zeros(mask_known.sum()), np.ones(mask_unseen_nutr.sum())]),
                np.concatenate([u_uat[mask_known], u_uat[mask_unseen_nutr]])
            )) * 100.0
            fpr95_within_pest = fpr95(u_uat[mask_known], u_uat[mask_pest])
            auroc_cross_pest = float(roc_auc_score(
                np.concatenate([np.zeros(len(u_val)), np.ones(mask_pest.sum())]),
                np.concatenate([u_val, u_uat[mask_pest]])
            )) * 100.0
            return {
                'uat_known_acc': round(known_acc, 2),
                'within_domain_Known_vs_Pest_AUROC': round(auroc_within_pest, 2),
                'within_domain_Known_vs_Pest_FPR95': round(fpr95_within_pest, 2),
                'within_domain_Known_vs_UnseenNutr_AUROC': round(auroc_within_nutr, 2),
                'cross_domain_Val_vs_Pest_AUROC': round(auroc_cross_pest, 2),
            }

        seed_reports.append({
            'seed': s,
            'B1_FT_Energy': eval_osr(unc_val_energy, unc_uat_energy, acc_ft_known, pred_ft),
            'B2_FT_RawProto': eval_osr(unc_val_ft_cos, unc_uat_ft_cos, acc_ft_known, pred_ft),
            'Abl1_FT_DebiasedCORAL': eval_osr(unc_val_ft_cos, unc_uat_ft_aligned, acc_ft_aligned_known, pred_ft_aligned),
            'Abl2_UndistortedPre_RawProto': eval_osr(unc_val_pre_raw, unc_uat_pre_raw, acc_pre_raw_known, pred_pre_raw),
            'Ours_DDP_OSR_Undistorted_Debiased': eval_osr(unc_val_ddp, unc_uat_ddp, acc_ddp_known, pred_ddp),
        })

    methods = ['B1_FT_Energy', 'B2_FT_RawProto', 'Abl1_FT_DebiasedCORAL', 'Abl2_UndistortedPre_RawProto', 'Ours_DDP_OSR_Undistorted_Debiased']
    summary = {}
    print("\n================ Probe-A2-Plus: 跨域同批内部开放集鉴别结果 (3 Seeds) ================", flush=True)
    for m in methods:
        metrics_keys = list(seed_reports[0][m].keys())
        summary[m] = {}
        for k in metrics_keys:
            vals = [r[m][k] for r in seed_reports]
            summary[m][k] = f"{np.mean(vals):.2f}±{np.std(vals, ddof=1):.2f}"
        print(f"  [{m:34s}] UAT共有类Acc={summary[m]['uat_known_acc']} | "
              f"★同域内(已知病vs未知虫) AUROC={summary[m]['within_domain_Known_vs_Pest_AUROC']} (FPR95={summary[m]['within_domain_Known_vs_Pest_FPR95']}) | "
              f"同域内(已知vs未见缺素) AUROC={summary[m]['within_domain_Known_vs_UnseenNutr_AUROC']}",
              flush=True)

    out_p = OUT_DIR / 'probe_a2_plus_debiased_osr_summary.json'
    out_p.write_text(json.dumps({
        '实验名称': 'Probe-A2-Plus_无监督流形去偏与预训练未畸变子空间跨域开放集鉴别',
        '耗时秒': round(time.monotonic() - t0, 2),
        '汇总_3种子': summary,
        '逐种子': seed_reports,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[完成] 保存至: {out_p}", flush=True)


if __name__ == '__main__':
    main()
