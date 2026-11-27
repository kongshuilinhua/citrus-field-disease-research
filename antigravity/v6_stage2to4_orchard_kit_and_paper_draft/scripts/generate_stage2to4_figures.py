# -*- coding: utf-8 -*-
"""
生成第二至四阶段综合论文配图 (fig4_external_with_negatives_and_huyou190_deployment.png)：
(a) 外部果园探索集 Zenodo 11148332 (519图, 3类 AP50-95 逐种子散点 + 均值，含 C5a 对照)；
(b) 室内灰底单叶集 CitrusUAT (500图, 5类 Macro-F1 逐种子散点 + 均值：诚实呈现田间检测器先验 -3.3pt 负结果及 w_det=0 归因验证)；
(c) 190 张 12MP (3072x4096) 衢州胡柚手机实拍端到端在线推理延迟与多拍同目标自洽率。
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

V6_DIR = Path(__file__).resolve().parents[1]
RES_JSON = V6_DIR / 'results' / 'stage2to4_external_and_orchard190_preunblind_complete.json'
FIG_DIR = V6_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    d = json.loads(RES_JSON.read_text(encoding='utf-8'))
    ext = d['module2_complete_external_matrix_3seeds']
    huyou = d['module3_huyou190_12mp_labelfree_deployment_audit']

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2), dpi=220)
    fig.patch.set_facecolor('white')

    # ---- Panel A: Zenodo 11148332 (519 orchard images, 3-class AP50-95) ----
    ax = axes[0]
    zen_s = ext['Zenodo_11148332_519imgs']['summary_per_seed_and_mean']
    zen_items = [
        ('B0_YOLOv8n', 'B0: YOLOv8n\n(0×ViT)', '#94a3b8'),
        ('B1_Full_DINOv2', 'B1: Full DINOv2\n(1×硬覆盖)', '#64748b'),
        ('A1_Single125_030_070', 'A1: 1.25×校准\n[0.30,0.70]', '#38bdf8'),
        ('G2_blur_Single125_BlurRing', 'G2_blur: 单刀模糊环\n(1×最简实现)', '#10b981'),
        ('A4b_FlipTTA_125_HFlip', 'A4b: 1.25×翻转TTA\n(2×同尺度)', '#f59e0b'),
        ('C3_equal_DualScale_035_035', 'C3_equal: 双尺度等权\n(2×主表零调参)', '#2563eb'),
        ('C5a_3Scale_PlainConcat', 'C5a: 三尺度MLP\n(3×参考上界)', '#8b5cf6'),
    ]
    y_pos = np.arange(len(zen_items))
    means_a = [zen_s[k]['3cls_AP50_95']['mean'] for k, _, _ in zen_items]
    colors_a = [c for _, _, c in zen_items]
    bars_a = ax.barh(y_pos, means_a, height=0.58, color=colors_a, alpha=0.86, edgecolor='#1e293b', linewidth=0.8)
    seed_markers = ['o', 's', '^']
    for yi, (k, _, _) in enumerate(zen_items):
        sds = zen_s[k]['3cls_AP50_95']['seeds']
        for si, sv in enumerate(sds):
            ax.scatter(sv, yi + (si - 1) * 0.13, marker=seed_markers[si], s=26, color='#0f172a', zorder=5, alpha=0.85)
        ax.text(max(sds + [means_a[yi]]) + 0.35, yi, f"{means_a[yi]:.2f}%", va='center', ha='left', fontsize=8.8, fontweight='bold', color='#0f172a')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl, _ in zen_items], fontsize=8.6)
    ax.invert_yaxis()
    xmin_a = max(0.0, min(min(zen_s[k]['3cls_AP50_95']['seeds']) for k, _, _ in zen_items) - 3.5)
    xmax_a = max(max(zen_s[k]['3cls_AP50_95']['seeds']) for k, _, _ in zen_items) + 4.8
    ax.set_xlim(xmin_a, xmax_a)
    ax.set_xlabel('3-Class COCO AP50–95 (%) [圆/方/三角 = S0/S1/S2]', fontsize=9.2)
    ax.set_title('(a) 外部真实果园集 Zenodo 11148332 (n=519)\n零丢框 3 类检测 AP50–95', fontsize=10.5, fontweight='bold', pad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.35)

    # ---- Panel B: CitrusUAT (500 indoor single-leaf OOD images — Honest Negative Result + Attribution) ----
    ax = axes[1]
    uat_s = ext['CitrusUAT_500imgs_IndoorSingleLeaf_OOD']['summary_per_seed_and_mean']
    uat_items = [
        ('B0_YOLOv8n', 'B0: YOLOv8n 检测头\n(田间先验严重OOD)', '#ef4444'),
        ('B1_Full_DINOv2', 'B1: 纯 DINOv2 (w_det=0)\n(室内单叶强基线)', '#0ea5e9'),
        ('A1_Single125_030_070', 'A1: 1.25×校准融合\n(含 w_det=0.30 负拖累)', '#f97316'),
        ('G2_blur_Single125_BlurRing', 'G2_blur: 模糊环融合\n(含 w_det=0.30 负拖累)', '#fb923c'),
        ('C3_equal_DualScale_035_035', 'C3_equal: 双尺度等权\n(含 w_det=0.30 负拖累)', '#f59e0b'),
        ('G2_blur_PureDino_wdet0', 'G2_blur (令 w_det=0)\n(剥离检测器先验验证)', '#10b981'),
        ('C3_PureDino_wdet0', 'C3 双尺度 (令 w_det=0)\n(剥离检测器先验验证)', '#2563eb'),
    ]
    y_b = np.arange(len(uat_items))
    means_b = [uat_s[k]['MacroF1_5cls']['mean'] for k, _, _ in uat_items]
    ax.barh(y_b, means_b, height=0.58, color=[c for _, _, c in uat_items], alpha=0.86, edgecolor='#1e293b', linewidth=0.8)
    for yi, (k, _, _) in enumerate(uat_items):
        sds = uat_s[k]['MacroF1_5cls']['seeds']
        for si, sv in enumerate(sds):
            ax.scatter(sv, yi + (si - 1) * 0.13, marker=seed_markers[si], s=26, color='#0f172a', zorder=5, alpha=0.85)
        ax.text(max(sds + [means_b[yi]]) + 0.6, yi, f"{means_b[yi]:.2f}%", va='center', ha='left', fontsize=8.8, fontweight='bold', color='#0f172a')

    ax.set_yticks(y_b)
    ax.set_yticklabels([lbl for _, lbl, _ in uat_items], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(max(0.0, min(means_b) - 8.0), max(means_b) + 9.5)
    ax.set_xlabel('5-Class Macro-F1 (%) [负结果诚实展示与归因]', fontsize=9.2)
    n_uat = ext['CitrusUAT_500imgs_IndoorSingleLeaf_OOD']['n_images']
    ax.set_title(f'(b) 室内灰底单叶集 CitrusUAT (n={n_uat})\n田间检测器先验负迁移与剥离验证', fontsize=10.5, fontweight='bold', pad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.35)

    # ---- Panel C: 190 HuYou 12MP Smartphone Real-Orchard Latency & Multi-Shot Consistency ----
    ax = axes[2]
    lat = huyou['same_session_12mp_latency_seed0']
    cons = huyou['multishot_same_object_consistency_3seeds']
    hy_items = [
        ('YOLOv8n', 'B0: YOLOv8n', '#94a3b8'),
        ('Full_DINOv2', 'B1: Full DINOv2', '#64748b'),
        ('C1_SingleScale_125', 'C1: 1.25×校准', '#38bdf8'),
        ('G2_blur_Single125_BlurRing', 'G2_blur: 单刀模糊环', '#10b981'),
        ('A4b_FlipTTA_125_HFlip', 'A4b: 翻转TTA', '#f59e0b'),
        ('C3_equal_DualScale_035_035', 'C3_equal: 双尺度等权', '#2563eb'),
    ]
    xs = [lat[k]['gpu_e2e_from_12mp_ram_mean_ms'] for k, _, _ in hy_items]
    ys = [cons[k]['multishot_top1_agree_pct']['mean'] for k, _, _ in hy_items]
    # Manual clean non-overlapping callout offsets (dx, dy, ha) for Panel C:
    callout_cfg = {
        'YOLOv8n': (5.0, 1.8, 'left'),
        'Full_DINOv2': (-5.0, -2.6, 'right'),
        'C1_SingleScale_125': (-5.5, 2.2, 'right'),
        'G2_blur_Single125_BlurRing': (5.5, -1.8, 'left'),
        'A4b_FlipTTA_125_HFlip': (5.5, 2.4, 'left'),
        'C3_equal_DualScale_035_035': (-5.5, 2.2, 'right'),
    }
    for idx_i, (k, lbl, col) in enumerate(hy_items):
        sds = cons[k]['multishot_top1_agree_pct']['seeds']
        for si, sv in enumerate(sds):
            ax.scatter(xs[idx_i], sv, marker=seed_markers[si], s=28, color=col, alpha=0.45, zorder=3)
        ax.scatter(xs[idx_i], ys[idx_i], s=115, color=col, edgecolor='#0f172a', linewidth=1.2, zorder=5)
        dx, dy, ha = callout_cfg[k]
        ax.annotate(
            f"{lbl}\n({xs[idx_i]:.1f}ms, {ys[idx_i]:.1f}%)",
            xy=(xs[idx_i], ys[idx_i]),
            xytext=(xs[idx_i] + dx, ys[idx_i] + dy),
            ha=ha, va='center', fontsize=8.2, fontweight='bold', color='#0f172a',
            arrowprops=dict(arrowstyle='-', color='#64748b', lw=0.8),
        )

    ax.set_xlim(min(xs) - 8.0, max(xs) + 18.0)
    ax.set_ylim(min(ys) - 6.5, min(102.0, max(ys) + 6.5))
    ax.set_xlabel('12MP (3072×4096) 手机原图端到端推理耗时 Mean (ms)', fontsize=9.2)
    ax.set_ylabel('同对象簇跨拍 Top-1 症状自洽率 (%)', fontsize=9.2)
    ax.set_title('(c) 衢州胡柚 12MP 手机实拍 (n=180 纯净自然图)\n端到端推理延迟 vs 同簇多拍预测自洽率', fontsize=10.5, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.35)

    plt.tight_layout(pad=1.6)
    out_p = FIG_DIR / 'fig4_external_with_negatives_and_huyou190_deployment.png'
    fig.savefig(out_p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"[配图生成完成] -> {out_p}", flush=True)


if __name__ == '__main__':
    main()
