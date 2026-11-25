# -*- coding: utf-8 -*-
"""
Probe-A2: 融合柑橘病理层级树的狄利克雷证据不确定性开放集诊断 (Hier-EDL) 先导验证
验证目标：
1. 源域保真：在 SciDB 14类症状分类 (val / testB / testC) 上不损失已知类判别精度；
2. 跨数据集零样本与开放集拒识：在未参与训练/调参的 CitrusUAT (953张/12类) 上：
   - 严格对应共有类 (Healthy->S2, Greasy_spot->S11) 保持高分类准确率；
   - 4类未知虫害 (Citrus_leafminer, Red_scale, Red_scale_sequelae, Texas_mite, N=322) 的未知拒识 AUROC / FPR@95TPR；
   - 3类未见营养缺素 (Fe, Mn, N, N=255) 被误判为高置信黄龙病典型症状 (S3 斑驳黄化) 的假阳性率，以及“大类归入黄化群、细类触发高不确定性复核”的层级诊断率。
3. 覆盖双骨干表征（DINOv2-ViT-S/14 与轻量 MobileNetV3-Large）× 3个随机种子 (seed 0, 1, 2)。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'

import json
import math
import time
import random
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
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(r'D:\新版柑橘')
BASE_CACHE = ROOT / '基线补齐_20261026'
OLD_LIST = ROOT / '本周科研_20261014' / '裁图清单.json'
CITRUS_UAT_DIR = ROOT / '复现_baseline' / 'field_test_CitrusUAT'
OUT_DIR = ROOT / '多路线创新筛选_20261124' / '探针A_选择性路由与证据拒识' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# SciDB 14类病理语义两级分组 (0-indexed):
# Group 0 (健康群): 0:S1 Healthy fruit, 1:S2 Healthy leaf
# Group 1 (黄化与类黄龙病生理异常群): 2:S3 Blotchy mottling, 3:S4 Red nose fruit, 4:S5 Zn, 5:S6 Vein yellowing, 6:S7 Uniform yellowing, 7:S8 Mg, 8:S9 B
# Group 2 (坏死与表生病斑群): 9:S10 Anthracnose, 10:S11 Greasy spot, 11:S14 Canker, 12:S13 Sooty mould, 13:S12 Algal spot
GROUPS = [
    [0, 1],
    [2, 3, 4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13],
]
FINE_TO_GROUP = np.zeros(14, dtype=np.int64)
for g_id, members in enumerate(GROUPS):
    for m in members:
        FINE_TO_GROUP[m] = g_id


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_citrus_uat_crops():
    """读取 CitrusUAT 全部 953 张原图并做统一 224x224 预处理。"""
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
            pil_im = Image.fromarray(rgb)
            imgs.append(np.asarray(tf(pil_im)).transpose(2, 0, 1))
            cls_names.append(cls_dir.name)
    return np.stack(imgs, axis=0), cls_names


@torch.no_grad()
def extract_backbone_embeddings(backbone_type: str, uat_imgs: np.ndarray, device: str = 'cuda:0'):
    """提取 SciDB (train/val/testB/testC) 与 CitrusUAT 在统一特征空间的 L2 归一化特征。"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)

    def run_array(arr_uint8: np.ndarray, model_fn):
        feats = []
        for start in range(0, len(arr_uint8), 128):
            chunk = torch.from_numpy(np.array(arr_uint8[start:start + 128])).to(device).float()
            x = (chunk / 255.0 - mean) / std
            z = F.normalize(model_fn(x).float(), dim=1)
            feats.append(z.cpu().numpy())
        return np.concatenate(feats, axis=0)

    if backbone_type == 'dinov2_lpft_backbone':
        meta = json.loads((ROOT / '方法推进_20261027' / '纯CLS对照' / '微调汇总.json').read_text(encoding='utf-8'))
        t_info = next(r for r in meta['重复结果'] if r['方法'] == 'LP-FT' and r['种子'] == 0)
        state = torch.load(t_info['权重文件'], map_location='cpu', weights_only=True)
        enc = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, img_size=224, num_classes=0).to(device).eval()
        enc.load_state_dict(state['骨干'])
        fn = lambda x: enc.forward_features(x)[:, 0]
    elif backbone_type == 'mobilenetv3_student_backbone':
        w_path = ROOT / '方法路线扩展_20261031' / '轻量像素分支_20261109' / '先导结果' / 'feature_seed0' / '最佳分类权重.pt'
        state = torch.load(w_path, map_location='cpu', weights_only=True)['模型状态']
        m = mobilenet_v3_large(weights=None)
        m.classifier[-1] = nn.Identity()
        proj = nn.Linear(1280, 384)
        enc_state = {k.replace('encoder.', ''): v for k, v in state.items() if k.startswith('encoder.')}
        proj_state = {k.replace('projector.', ''): v for k, v in state.items() if k.startswith('projector.')}
        m.load_state_dict(enc_state)
        proj.load_state_dict(proj_state)
        m = m.to(device).eval()
        proj = proj.to(device).eval()
        fn = lambda x: proj(m(x))
    else:
        raise ValueError(backbone_type)

    splits = {}
    for sp in ['train', 'val', 'testB', 'testC']:
        arr = np.load(BASE_CACHE / '图像缓存' / f'{sp}.npy', mmap_mode='r')
        splits[sp] = run_array(arr, fn)
    uat_feats = run_array(uat_imgs, fn)
    return splits, uat_feats


class DiagnosticHead(nn.Module):
    """两层非线性残差诊断头（支持 CE / Flat-EDL / Hier-EDL 公平同参数量对比）。"""
    def __init__(self, in_dim: int = 384, hidden_dim: int = 256, num_classes: int = 14, mode: str = 'ce'):
        super().__init__()
        self.mode = mode
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc_fine = nn.Linear(hidden_dim, num_classes)
        if mode == 'hier_edl':
            self.fc_group = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        h = F.gelu(self.ln(self.fc1(x)))
        out_fine = self.fc_fine(h)
        if self.mode == 'hier_edl':
            out_group = self.fc_group(h)
            return h, out_fine, out_group
        return h, out_fine, None


def kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """计算 Dirichlet(alpha) 相对均匀先验 Dirichlet(1,...,1) 的解析 KL 散度。"""
    K = alpha.shape[1]
    S = alpha.sum(dim=1, keepdim=True)
    ones = torch.ones_like(alpha)
    lnB_alpha = torch.lgamma(S) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    lnB_uni = torch.lgamma(torch.tensor(float(K), device=alpha.device))
    digamma_term = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1, keepdim=True)
    return (lnB_alpha - lnB_uni + digamma_term).squeeze(1)


def edl_digamma_loss(evidence: torch.Tensor, target: torch.Tensor, epoch: int, kl_coef: float = 0.2):
    K = evidence.shape[1]
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    y_onehot = F.one_hot(target, num_classes=K).float()
    # Bayes risk of cross-entropy under Dirichlet
    ace = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
    # Remove true class evidence before penalizing spurious evidence
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    anneal = min(1.0, epoch / 15.0)
    kl = kl_dirichlet_uniform(alpha_tilde)
    return (ace + anneal * kl_coef * kl).mean()


def fpr_at_95_tpr(id_Continue_scores: np.ndarray, ood_uncertainty_scores: np.ndarray) -> float:
    """以 ID 样本不确定性为负类(低)、OOD 样本不确定性为正类(高)，计算在 95% ID 保留率(TPR=0.95)下的 OOD 漏放率(FPR)。"""
    # Threshold tau on ID uncertainty such that 95% of ID samples have uncertainty <= tau
    tau = np.quantile(id_Continue_scores, 0.95)
    # FPR = fraction of OOD samples that mistakenly have uncertainty <= tau (accepted as known!)
    return float((ood_uncertainty_scores <= tau).mean())


def train_and_eval_head(mode_name: str, splits_x: dict, splits_y: dict, uat_x: np.ndarray, uat_cls: list, seed: int, device='cuda:0'):
    set_seed(seed)
    base_mode = 'ce' if mode_name in ['ce_msp', 'ce_energy'] else ('flat_edl' if mode_name == 'flat_edl' else 'hier_edl')
    model = DiagnosticHead(in_dim=splits_x['train'].shape[1], hidden_dim=256, num_classes=14, mode=base_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

    x_tr = torch.from_numpy(splits_x['train']).to(device)
    y_tr = torch.from_numpy(splits_y['train']).to(device)
    g_tr = torch.from_numpy(FINE_TO_GROUP[splits_y['train']]).to(device)

    x_val = torch.from_numpy(splits_x['val']).to(device)
    y_val_np = splits_y['val']

    # Class prototype centers in latent h space for prototype-modulated evidence in Hier-EDL
    best_val_f1 = -1.0
    best_state = None

    for ep in range(1, 61):
        model.train()
        perm = torch.randperm(len(x_tr), device=device)
        for ids in perm.split(256):
            opt.zero_grad(set_to_none=True)
            xb, yb, gb = x_tr[ids], y_tr[ids], g_tr[ids]
            h, out_fine, out_group = model(xb)

            if base_mode == 'ce':
                loss = F.cross_entropy(out_fine, yb)
            elif base_mode == 'flat_edl':
                ev_fine = F.softplus(out_fine)
                loss = edl_digamma_loss(ev_fine, yb, ep, kl_coef=0.2)
            elif base_mode == 'hier_edl':
                ev_fine = F.softplus(out_fine)
                ev_group = F.softplus(out_group)
                loss_fine = edl_digamma_loss(ev_fine, yb, ep, kl_coef=0.25)
                loss_group = edl_digamma_loss(ev_group, gb, ep, kl_coef=0.15)

                # 层级证据一致性约束：大类总信念质量与子类信念之和对齐
                agg_group_ev = torch.stack([ev_fine[:, GROUPS[g]].sum(dim=1) for g in range(3)], dim=1)
                p_group_branch = (ev_group + 1.0) / (ev_group.sum(dim=1, keepdim=True) + 3.0)
                p_group_agg = (agg_group_ev + 1.0) / (agg_group_ev.sum(dim=1, keepdim=True) + 3.0)
                loss_cons = F.mse_loss(p_group_branch, p_group_agg.detach()) + 0.5 * F.mse_loss(p_group_agg, p_group_branch.detach())

                # 黄化群内部硬负样本边界约束（抑制同属黄化群的错误子类虚高证据）
                in_yellow = (gb == 1)
                if in_yellow.any():
                    yf_ev = ev_fine[in_yellow][:, GROUPS[1]]  # (N_y, 7)
                    y_local = yb[in_yellow] - 2               # 0..6
                    mask_wrong_sib = 1.0 - F.one_hot(y_local, num_classes=7).float()
                    loss_sib = ((yf_ev * mask_wrong_sib) ** 2).mean() * 0.02
                else:
                    loss_sib = 0.0

                loss = loss_fine + 0.5 * loss_group + 0.5 * loss_cons + loss_sib
            loss.backward()
            opt.step()
        sched.step()

        # Select checkpoint strictly by SciDB val 14-class Macro-F1
        model.eval()
        with torch.no_grad():
            _, vf, _ = model(x_val)
            v_pred = vf.argmax(dim=1).cpu().numpy()
            vf1 = float(f1_score(y_val_np, v_pred, average='macro'))
            if vf1 > best_val_f1:
                best_val_f1 = vf1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # Fit latent Mahalanobis density modulator on train h (for Hier-EDL epistemic density awareness!)
    with torch.no_grad():
        h_tr, _, _ = model(x_tr)
        h_tr_np = F.normalize(h_tr, dim=1).cpu().numpy()
    tr_means = np.zeros((14, h_tr_np.shape[1]), dtype=np.float64)
    centered = []
    for c in range(14):
        m_c = (splits_y['train'] == c)
        Xc = h_tr_np[m_c].astype(np.float64)
        tr_means[c] = Xc.mean(axis=0)
        centered.append(Xc - tr_means[c])
    cov = (np.concatenate(centered, axis=0).T @ np.concatenate(centered, axis=0)) / max(1, len(h_tr_np) - 1)
    prec = np.linalg.inv(cov + 1e-2 * np.eye(h_tr_np.shape[1]))

    # Compute val reference radial distance scale (strictly on train/val, zero UAT leakage)
    with torch.no_grad():
        h_val_t, _, _ = model(x_val)
        h_val_np = F.normalize(h_val_t, dim=1).cpu().numpy().astype(np.float64)
    val_dmin = np.min(np.stack([np.sqrt(np.maximum(0.0, np.sum(((h_val_np - tr_means[c]) @ prec) * (h_val_np - tr_means[c]), axis=1))) for c in range(14)], axis=1), axis=1)
    sigma_scale = float(np.median(val_dmin))

    @torch.no_grad()
    def infer_with_uncertainty(feat_np: np.ndarray):
        xt = torch.from_numpy(feat_np).to(device)
        h, out_fine, out_group = model(xt)
        pred_fine = out_fine.argmax(dim=1).cpu().numpy()

        if mode_name == 'ce_msp':
            probs = out_fine.softmax(dim=1).cpu().numpy()
            unc_fine = 1.0 - np.max(probs, axis=1)
            unc_group = np.zeros_like(unc_fine)
            pred_group = FINE_TO_GROUP[pred_fine]
        elif mode_name == 'ce_energy':
            unc_fine = -torch.logsumexp(out_fine, dim=1).cpu().numpy()
            unc_group = np.zeros_like(unc_fine)
            pred_group = FINE_TO_GROUP[pred_fine]
        elif mode_name == 'flat_edl':
            ev = F.softplus(out_fine)
            S = (ev + 1.0).sum(dim=1)
            unc_fine = (14.0 / S).cpu().numpy()  # Vacuity u = K / S
            unc_group = np.zeros_like(unc_fine)
            pred_group = FINE_TO_GROUP[pred_fine]
        elif mode_name == 'hier_edl':
            # Density-modulated Hierarchical Dirichlet Evidence (PostNet/Charpentier-inspired density Attenuation on unseen manifold)
            h_np = F.normalize(h, dim=1).cpu().numpy().astype(np.float64)
            d_all = np.stack([np.sqrt(np.maximum(0.0, np.sum(((h_np - tr_means[c]) @ prec) * (h_np - tr_means[c]), axis=1))) for c in range(14)], axis=1)
            d_min = np.min(d_all, axis=1)
            # Smooth attenuation factor in (0, 1]: near-1 on ID manifold, decays smoothly outside manifold
            attenuation = np.exp(-0.5 * np.maximum(0.0, (d_min / max(1e-6, sigma_scale)) - 1.0) ** 2)
            att_t = torch.from_numpy(attenuation).to(device).float().unsqueeze(1)

            ev_fine = F.softplus(out_fine) * att_t
            S_fine = (ev_fine + 1.0).sum(dim=1)
            unc_fine = (14.0 / S_fine).cpu().numpy()

            ev_group = F.softplus(out_group)
            S_group = (ev_group + 1.0).sum(dim=1)
            unc_group = (3.0 / S_group).cpu().numpy()
            pred_group = ev_group.argmax(dim=1).cpu().numpy()
            # Combine fine vacuity with hierarchical conflict |p_group - agg_p_group|
            probs_fine = (ev_fine + 1.0) / S_fine.unsqueeze(1)
            agg_g = torch.stack([probs_fine[:, GROUPS[g]].sum(dim=1) for g in range(3)], dim=1)
            probs_g = (ev_group + 1.0) / S_group.unsqueeze(1)
            hier_conflict = torch.abs(probs_g - agg_g).sum(dim=1).cpu().numpy()
            unc_fine = unc_fine + 0.5 * hier_conflict

        return pred_fine, pred_group, unc_fine, unc_group

    # 1. SciDB val / testB / testC accuracy & Macro-F1
    scidb_metrics = {}
    val_unc = None
    for sp in ['val', 'testB', 'testC']:
        pf, _, uf, _ = infer_with_uncertainty(splits_x[sp])
        acc = float((pf == splits_y[sp]).mean()) * 100.0
        mf1 = float(f1_score(splits_y[sp], pf, average='macro')) * 100.0
        scidb_metrics[sp] = {'acc': round(acc, 2), 'macro_f1': round(mf1, 2)}
        if sp == 'val':
            val_unc = uf

    # Calibrate 95%-TPR ID retention threshold STRICTLY on SciDB val!
    tau_95 = float(np.quantile(val_unc, 0.95))

    # 2. Zero-shot CitrusUAT evaluation
    uat_pf, uat_pg, uat_uf, uat_ug = infer_with_uncertainty(uat_x)
    uat_cls_arr = np.array(uat_cls)

    # Known compatible subset: Healthy -> S2 (1), Greasy_spot -> S11 (10)
    mask_healthy = (uat_cls_arr == 'Healthy')
    mask_greasy = (uat_cls_arr == 'Greasy_spot')
    acc_healthy = float((uat_pf[mask_healthy] == 1).mean()) * 100.0
    acc_greasy = float((uat_pf[mask_greasy] == 10).mean()) * 100.0
    mask_known_compat = mask_healthy | mask_greasy
    y_known = np.where(uat_cls_arr[mask_known_compat] == 'Healthy', 1, 10)
    acc_known_all = float((uat_pf[mask_known_compat] == y_known).mean()) * 100.0

    # OOD Set 1: 4 Unseen Pest classes (Leafminer, Red_scale, Red_scale_sequelae, Texas_mite)
    pest_classes = {'Citrus_leafminer', 'Red_scale', 'Red_scale_sequelae', 'Texas_mite'}
    mask_pest = np.array([c in pest_classes for c in uat_cls])
    # AUROC: SciDB val (ID=0) vs CitrusUAT Pests (OOD=1), AND CitrusUAT Known (ID=0) vs CitrusUAT Pests (OOD=1)
    auroc_val_vs_pest = float(roc_auc_score(
        np.concatenate([np.zeros(len(val_unc)), np.ones(mask_pest.sum())]),
        np.concatenate([val_unc, uat_uf[mask_pest]])
    )) * 100.0
    auroc_uat_known_vs_pest = float(roc_auc_score(
        np.concatenate([np.zeros(mask_known_compat.sum()), np.ones(mask_pest.sum())]),
        np.concatenate([uat_uf[mask_known_compat], uat_uf[mask_pest]])
    )) * 100.0
    fpr95_val_vs_pest = fpr_at_95_tpr(val_unc, uat_uf[mask_pest]) * 100.0

    # OOD Set 2: 3 Unseen Nutrient Deficiencies (Fe, Mn, N)
    unseen_nutr = {'Fe', 'Mn', 'N'}
    mask_nutr = np.array([c in unseen_nutr for c in uat_cls])
    # Without rejection: how many Fe/Mn/N are predicted as S3 Blotchy mottling (2) or HLB-suspect yellowing {S3(2), S6(5), S7(6)}?
    raw_hlb_mottling_fp = float((uat_pf[mask_nutr] == 2).mean()) * 100.0
    raw_hlb_suspect_fp = float(np.isin(uat_pf[mask_nutr], [2, 5, 6]).mean()) * 100.0
    # With val-calibrated tau_95 rejection: fraction of Fe/Mn/N that STILL slip through (uf <= tau_95) AND get misdiagnosed as HLB-suspect {2,5,6}!
    accepted_nutr = (uat_uf[mask_nutr] <= tau_95)
    calibrated_hlb_suspect_fp = float((accepted_nutr & np.isin(uat_pf[mask_nutr], [2, 5, 6])).mean()) * 100.0
    nutr_rejection_rate = float((~accepted_nutr).mean()) * 100.0
    # Group-level chlorosis recognition rate on Fe/Mn/N (correctly routed to Group 1 Chlorosis!)
    nutr_group_chlorosis_rate = float((uat_pg[mask_nutr] == 1).mean()) * 100.0
    hier_safe_triage_rate = float(((uat_pg[mask_nutr] == 1) & (~accepted_nutr)).mean()) * 100.0

    return {
        'seed': seed,
        'scidb': scidb_metrics,
        'citrus_uat': {
            'known_compat_acc': round(acc_known_all, 2),
            'healthy_acc': round(acc_healthy, 2),
            'greasy_spot_acc': round(acc_greasy, 2),
            'pest_ood_auroc_vs_scidb_val': round(auroc_val_vs_pest, 2),
            'pest_ood_auroc_vs_uat_known': round(auroc_uat_known_vs_pest, 2),
            'pest_ood_fpr95_vs_scidb_val': round(fpr95_val_vs_pest, 2),
            'unseen_FeMnN_raw_S3_fp_pct': round(raw_hlb_mottling_fp, 2),
            'unseen_FeMnN_raw_HLBsuspect_fp_pct': round(raw_hlb_suspect_fp, 2),
            'unseen_FeMnN_calibrated_HLBsuspect_fp_pct': round(calibrated_hlb_suspect_fp, 2),
            'unseen_FeMnN_fine_reject_rate_pct': round(nutr_rejection_rate, 2),
            'unseen_FeMnN_coarse_yellowing_hit_pct': round(nutr_group_chlorosis_rate, 2),
            'unseen_FeMnN_hier_safe_triage_pct': round(hier_safe_triage_rate, 2),
        }
    }


def main():
    t0 = time.monotonic()
    recs = json.loads(OLD_LIST.read_text(encoding='utf-8-sig'))
    splits_y = {sp: np.array([r['类别'] - 1 for r in recs[sp]], dtype=np.int64) for sp in ['train', 'val', 'testB', 'testC']}

    print("[1/3] 读取 CitrusUAT 953 张原图...", flush=True)
    uat_imgs, uat_cls = load_citrus_uat_crops()
    print(f"  CitrusUAT 加载完成: {len(uat_imgs)} 张, 类别分布={ {c: uat_cls.count(c) for c in sorted(set(uat_cls))} }", flush=True)

    full_summary = {}
    for backbone in ['dinov2_lpft_backbone', 'mobilenetv3_student_backbone']:
        print(f"\n================ 提取特征骨干: {backbone} ================", flush=True)
        splits_x, uat_x = extract_backbone_embeddings(backbone, uat_imgs, 'cuda:0')
        backbone_res = {}
        for mode in ['ce_msp', 'ce_energy', 'flat_edl', 'hier_edl']:
            seed_runs = [train_and_eval_head(mode, splits_x, splits_y, uat_x, uat_cls, s, 'cuda:0') for s in [0, 1, 2]]
            # Aggregate mean ± std
            def m_std(getter):
                vals = [getter(r) for r in seed_runs]
                return f"{np.mean(vals):.2f}±{np.std(vals, ddof=1):.2f}", round(float(np.mean(vals)), 2), round(float(np.std(vals, ddof=1)), 2)

            agg = {
                'val_macro_f1': m_std(lambda r: r['scidb']['val']['macro_f1']),
                'testB_macro_f1': m_std(lambda r: r['scidb']['testB']['macro_f1']),
                'testC_macro_f1': m_std(lambda r: r['scidb']['testC']['macro_f1']),
                'uat_known_acc': m_std(lambda r: r['citrus_uat']['known_compat_acc']),
                'pest_auroc_vs_val': m_std(lambda r: r['citrus_uat']['pest_ood_auroc_vs_scidb_val']),
                'pest_auroc_vs_uat': m_std(lambda r: r['citrus_uat']['pest_ood_auroc_vs_uat_known']),
                'pest_fpr95': m_std(lambda r: r['citrus_uat']['pest_ood_fpr95_vs_scidb_val']),
                'FeMnN_raw_HLBsuspect_fp': m_std(lambda r: r['citrus_uat']['unseen_FeMnN_raw_HLBsuspect_fp_pct']),
                'FeMnN_calibrated_HLBsuspect_fp': m_std(lambda r: r['citrus_uat']['unseen_FeMnN_calibrated_HLBsuspect_fp_pct']),
                'FeMnN_hier_safe_triage': m_std(lambda r: r['citrus_uat']['unseen_FeMnN_hier_safe_triage_pct']),
                'seeds': seed_runs,
            }
            backbone_res[mode] = agg
            print(f"  [{mode:10s}] SciDB val F1={agg['val_macro_f1'][0]} | B F1={agg['testB_macro_f1'][0]} | C F1={agg['testC_macro_f1'][0]} | "
                  f"UAT共有类Acc={agg['uat_known_acc'][0]} | 虫害AUROC={agg['pest_auroc_vs_val'][0]}(同域内{agg['pest_auroc_vs_uat'][0]}) FPR95={agg['pest_fpr95'][0]} | "
                  f"未见Fe/Mn/N误判类HLB: 原{agg['FeMnN_raw_HLBsuspect_fp'][0]}% -> 校准后{agg['FeMnN_calibrated_HLBsuspect_fp'][0]}% (层级安全分流={agg['FeMnN_hier_safe_triage'][0]}%)",
                  flush=True)
        full_summary[backbone] = backbone_res

    out_path = OUT_DIR / 'probe_a2_hierarchical_edl_summary.json'
    out_path.write_text(json.dumps({
        '实验名称': 'Probe-A2_病理层级狄利克雷证据开放集诊断',
        '耗时秒': round(time.monotonic() - t0, 2),
        '结果': full_summary,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[完成] Probe-A2 结果已保存至: {out_path}", flush=True)


if __name__ == '__main__':
    main()
