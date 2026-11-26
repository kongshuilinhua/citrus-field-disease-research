# -*- coding: utf-8 -*-
"""
绘制 Stage 2 & Stage 2b 全部无重叠论文级图表：
1. fig1_c3_controlled_scale_ablation_and_bootstrap.png (受控尺度消融含等权对照 & 补齐配对 Bootstrap 森林图)
2. fig2_c3_fair_end2end_latency_and_pareto.png (统一同会话端到端公平测速，补齐 A4b 与 C1，彻底剔除跨协议混比)
3. fig3_g1_to_g4_gray_and_blur_ring_mechanism.png (G1–G4 灰环 + G2_blur/G4_blur 高斯模糊外环机制实验与双重差分 DiD 森林图)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

V5_DIR = Path(r'D:\新版柑橘\citrus-field-disease-research\antigravity\v5_c3_main_method_and_ablation')
STAGE2_JSON = V5_DIR / 'results' / 'stage2_c3_controlled_scale_ablation_complete.json'
STAGE2B_JSON = V5_DIR / 'results' / 'stage2b_gray_ring_mechanism_and_a4b_speed.json'
FIG_DIR = V5_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_fig1_scale_ablation_and_bootstrap(data: dict):
    ab = data['ablation_metrics_3seed']
    bt = data['paired_bootstrap_1000_results']

    items = [
        ('B0_Native_YOLOv8n', 'B0: YOLOv8n\n(单阶段基线)'),
        ('B1_Full_DINOv2_125_Hard', 'B1: Full DINOv2\n(1.25x 硬覆盖)'),
        ('A1_Single_125_Iso30', 'A1: 1.25x 单尺度\n(标准视角, 0.3/0.7)'),
        ('A2_Single_082_Iso30', 'A2: 0.82x 单尺度\n(紧致视角, 0.3/0.7)'),
        ('A4a_CloseDual_115_135_Iso30_ValGrid', 'A4a: 1.15x+1.35x\n(相近双裁剪TTA)'),
        ('A4b_FlipTTA_125_HFlip_Iso30_ValGrid', 'A4b: 1.25x+翻转\n(常规双视角TTA)'),
        ('A3_Dual_082_125_Iso30_Equal', 'C3-Eq: 0.82x+1.25x\n(零调参等权 0.35/0.35)'),
        ('A3_C3_Locked_082_125_Main', 'A3/C3: 0.82x+1.25x\n(唯一冻结主方法)'),
        ('A5_TriScale_082_125_250_Iso30_ValGrid', 'A5: +2.50x 三尺度\n(追加大范围环境)'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.2, 7.0), gridspec_kw={'width_ratios': [1.30, 1.08]})
    fig.subplots_adjust(wspace=0.43, bottom=0.23, top=0.85)

    x = np.arange(len(items))
    w = 0.38
    b_m = [ab[k]['testB']['AP50_95_mean'] for k, _ in items]
    b_s = [ab[k]['testB']['AP50_95_std'] for k, _ in items]
    c_m = [ab[k]['testC']['AP50_95_mean'] for k, _ in items]
    c_s = [ab[k]['testC']['AP50_95_std'] for k, _ in items]

    bars_b = ax1.bar(x - w / 2, b_m, w, yerr=b_s, capsize=3.0, color='#3B82F6', alpha=0.88,
                     error_kw={'lw': 1.15, 'ecolor': '#1E3A8A'}, label='SciDB testB (自然田间复杂背景)')
    bars_c = ax1.bar(x + w / 2, c_m, w, yerr=c_s, capsize=3.0, color='#10B981', alpha=0.88,
                     error_kw={'lw': 1.15, 'ecolor': '#064E3B'}, label='SciDB testC (均匀白板异质背景)')

    bars_b[7].set_edgecolor('#B91C1C'); bars_b[7].set_linewidth(2.2)
    bars_c[7].set_edgecolor('#B91C1C'); bars_c[7].set_linewidth(2.2)

    for rect, val, sd in zip(bars_b, b_m, b_s):
        ax1.text(rect.get_x() + rect.get_width() / 2, val + sd + 0.42, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=7.5, color='#1E3A8A', fontweight='bold')
    for rect, val, sd in zip(bars_c, c_m, c_s):
        ax1.text(rect.get_x() + rect.get_width() / 2, val + sd + 0.42, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=7.5, color='#065F46', fontweight='bold')

    ax1.set_ylim(63, 93.2)
    ax1.set_ylabel('COCO AP50-95 (%, 3随机种子均值±标准差)', fontsize=10.5)
    ax1.set_title('(a) 严格固定 w_det=0.30 受控尺度消融 (含冻结 C3 与零调参等权对照)', fontsize=11.0, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in items], fontsize=8.3, rotation=24, ha='right')
    ax1.grid(axis='y', ls=':', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=9.1, frameon=True)

    forest_pairs = [
        ('A3_C3_Locked_082_125_Main__minus__A1_Single_125_Iso30', 'C3 - A1 (1.25x 标准单尺度)'),
        ('A3_C3_Locked_082_125_Main__minus__A2_Single_082_Iso30', 'C3 - A2 (0.82x 紧致单尺度)'),
        ('A3_C3_Locked_082_125_Main__minus__A4a_CloseDual_115_135_Iso30_ValGrid', 'C3 - A4a (1.15x+1.35x 相近TTA)'),
        ('A3_C3_Locked_082_125_Main__minus__A4b_FlipTTA_125_HFlip_Iso30_ValGrid', 'C3 - A4b (1.25x+翻转 常规TTA)'),
        ('A3_C3_Locked_082_125_Main__minus__A3_Dual_082_125_Iso30_Equal', 'C3 - C3_equal (冻结 vs 等权 0.35/0.35)'),
        ('A5_TriScale_082_125_250_Iso30_ValGrid__minus__A3_C3_Locked_082_125_Main', 'A5 (+2.50x 环境视角) - C3'),
    ]
    y_pos = np.arange(len(forest_pairs))[::-1]
    off = 0.17

    ax2.axvline(0.0, color='#DC2626', ls='--', lw=1.4, alpha=0.85, label='零差异基准线 (ΔAP = 0)')
    for yi, (pk, plbl) in zip(y_pos, forest_pairs):
        rb = bt['testB'][pk]['AP50_95']
        rc = bt['testC'][pk]['AP50_95']
        mb, (lb, hb), pb = rb['mean_diff_3seed'], rb['ci95_3seed'], rb['p_value_two_sided']
        mc, (lc, hc), pc = rc['mean_diff_3seed'], rc['ci95_3seed'], rc['p_value_two_sided']

        ax2.errorbar(mb, yi + off, xerr=[[mb - lb], [hb - mb]], fmt='o', color='#2563EB', capsize=3.8, lw=1.7, ms=6.0,
                     label='SciDB testB (95% CI)' if yi == y_pos[0] else None)
        ax2.errorbar(mc, yi - off, xerr=[[mc - lc], [hc - mc]], fmt='s', color='#059669', capsize=3.8, lw=1.7, ms=6.0,
                     label='SciDB testC (95% CI)' if yi == y_pos[0] else None)

        ax2.text(hb + 0.08, yi + off, f'{mb:+.2f}% (p={pb:.3f})', va='center', ha='left', fontsize=7.9, color='#1E40AF', fontweight='bold')
        ax2.text(hc + 0.08, yi - off, f'{mc:+.2f}% (p={pc:.3f})', va='center', ha='left', fontsize=7.9, color='#065F46', fontweight='bold')

    all_hi = [max(bt['testB'][pk]['AP50_95']['ci95_3seed'][1], bt['testC'][pk]['AP50_95']['ci95_3seed'][1]) for pk, _ in forest_pairs]
    all_lo = [min(bt['testB'][pk]['AP50_95']['ci95_3seed'][0], bt['testC'][pk]['AP50_95']['ci95_3seed'][0]) for pk, _ in forest_pairs]
    ax2.set_xlim(min(-1.3, min(all_lo) - 0.5), max(all_hi) + 2.35)
    ax2.set_ylim(-0.65, len(forest_pairs) - 0.20)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([lbl for _, lbl in forest_pairs], fontsize=8.9)
    ax2.set_xlabel('3种子同步逐图配对 Bootstrap ΔAP50-95 (%) 及 95% 置信区间', fontsize=10.2)
    ax2.set_title('(b) 等检测器权重 (w_det=0.30) 关键假设配对检验森林图 (B=1,000)', fontsize=11.0, fontweight='bold', pad=12)
    ax2.grid(axis='x', ls=':', alpha=0.5)
    ax2.legend(loc='lower right', fontsize=8.7, frameon=True)

    fig.suptitle('唯一主方法 C3 (0.82× + 1.25× 双尺度校准后验融合) 受控尺度消融与完备配对 Bootstrap 检验', fontsize=13.4, fontweight='bold', y=0.96)
    out_p = FIG_DIR / 'fig1_c3_controlled_scale_ablation_and_bootstrap.png'
    fig.savefig(out_p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_p}")


def plot_fig2_fair_latency_profile(data: dict):
    lat = data['fair_end2end_latency_rtx4060ti_fp32']
    ab = data['ablation_metrics_3seed']

    modes = [
        ('B0_Native_YOLOv8n', 'B0: YOLOv8n', 'B0_Native_YOLOv8n', '#94A3B8'),
        ('B1_Full_DINOv2_125_Hard', 'B1: Full DINOv2', 'B1_Full_DINOv2_125_Hard', '#64748B'),
        ('C1_Single_125_035_065', 'C1: 1.25x (0.35/0.65)', 'C1_Historical_125_035_065', '#2563EB'),
        ('A1_Single_125', 'A1: 1.25x (0.30/0.70)', 'A1_Single_125_Iso30', '#3B82F6'),
        ('A2_Single_082', 'A2: 0.82x 紧致单尺度', 'A2_Single_082_Iso30', '#0EA5E9'),
        ('A4a_CloseDual_115_135', 'A4a: 1.15x+1.35x TTA', 'A4a_CloseDual_115_135_Iso30_ValGrid', '#F59E0B'),
        ('A4b_FlipTTA_125_HFlip', 'A4b: 1.25x+翻转 TTA', 'A4b_FlipTTA_125_HFlip_Iso30_ValGrid', '#EC4899'),
        ('A3_Dual_082_125_C3_Main', 'A3/C3: 0.82x+1.25x 主方法', 'A3_C3_Locked_082_125_Main', '#10B981'),
        ('A5_TriScale_082_125_250', 'A5: +2.50x 三尺度', 'A5_TriScale_082_125_250_Iso30_ValGrid', '#8B5CF6'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.4, 6.9), gridspec_kw={'width_ratios': [1.18, 1.05]})
    fig.subplots_adjust(wspace=0.25, bottom=0.23, top=0.85)

    x = np.arange(len(modes))
    w = 0.26
    m_val = [lat[mk]['val_250imgs']['mean_ms'] for mk, _, _, _ in modes]
    p95_val = [lat[mk]['val_250imgs']['p95_ms'] for mk, _, _, _ in modes]
    mt_val = [lat[mk]['val_250imgs']['multi_target_ge3_mean_ms'] for mk, _, _, _ in modes]

    r1 = ax1.bar(x - w, m_val, w, color='#3B82F6', alpha=0.90, label='全集平均延迟 Mean (ms)')
    r2 = ax1.bar(x, mt_val, w, color='#F59E0B', alpha=0.90, label='多目标图像 (≥3框) 平均延迟 (ms)')
    r3 = ax1.bar(x + w, p95_val, w, color='#EF4444', alpha=0.85, label='P95 尾部延迟 (ms)')

    for rects in (r1, r2, r3):
        for r in rects:
            v = r.get_height()
            ax1.text(r.get_x() + r.get_width() / 2, v + 0.55, f'{v:.1f}', ha='center', va='bottom', fontsize=6.8, fontweight='bold', rotation=35)

    ax1.set_ylim(0, max(p95_val) * 1.26)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl, _, _ in modes], fontsize=8.3, rotation=24, ha='right')
    ax1.set_ylabel('同会话端到端单图总推理耗时 (ms, RTX 4060 Ti FP32)', fontsize=10.3)
    ax1.set_title('(a) 含 A4b 的同会话公平延迟剖面 (SciDB val: 解码+检测+裁剪+分类+融合+NMS)', fontsize=10.7, fontweight='bold', pad=11)
    ax1.grid(axis='y', ls=':', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=8.8, frameon=True)

    # Right panel: Accuracy on complex field testB vs testC end-to-end latency with clean repelled callouts
    pts = []
    for mk, mlbl, ab_k, col in modes:
        ms_c = lat[mk]['testC_all602imgs']['mean_ms']
        ap_b = ab[ab_k]['testB']['AP50_95_mean']
        ap_c = ab[ab_k]['testC']['AP50_95_mean']
        pts.append((mk, mlbl, ms_c, ap_b, ap_c, col))

    ms_vals = [p[2] for p in pts]
    apb_vals = [p[3] for p in pts]
    ax2.set_xlim(min(ms_vals) - 2.0, max(ms_vals) + 12.6)
    y_min, y_max = min(apb_vals) - 1.5, max(apb_vals) + 2.4
    ax2.set_ylim(y_min, y_max)

    # Sort points by y (testB AP) and assign evenly spaced non-overlapping label y-coordinates on right/left sides
    pts_sorted = sorted(pts, key=lambda z: z[3])
    label_ys = np.linspace(y_min + 0.7, y_max - 0.6, len(pts_sorted))
    for (mk, mlbl, ms_c, ap_b, ap_c, col), ly in zip(pts_sorted, label_ys):
        is_c3 = 'C3_Main' in mk
        mk_sz = 150 if is_c3 else 88
        edge = '#B91C1C' if is_c3 else 'black'
        ax2.scatter(ms_c, ap_b, s=mk_sz, color=col, edgecolors=edge, linewidths=2.0 if is_c3 else 1.0, zorder=5)
        lx = max(ms_vals) + 0.7 if ms_c >= 14.0 else ms_c + 1.2
        ax2.annotate(
            f"{mlbl}: {ms_c:.2f}ms | B={ap_b:.2f}% / C={ap_c:.2f}%",
            xy=(ms_c, ap_b), xytext=(lx, ly),
            ha='left', va='center', fontsize=7.8,
            fontweight='bold' if is_c3 else 'normal',
            arrowprops=dict(arrowstyle='->', color=col, lw=1.0, alpha=0.75),
            bbox=dict(boxstyle='round,pad=0.22', fc='#FEF2F2' if is_c3 else 'white', ec=edge if is_c3 else col, alpha=0.92),
        )

    ax2.set_xlabel('同会话端到端单图平均延迟 (ms, SciDB testC 602图实测)', fontsize=10.3)
    ax2.set_ylabel('复杂自然冠层 SciDB testB AP50-95 (%, 括号附 testC)', fontsize=10.3)
    ax2.set_title('(b) 复杂背景精度 vs 端到端延迟权衡 (含 A4b 与 C1 同协议对比)', fontsize=10.7, fontweight='bold', pad=11)
    ax2.grid(True, ls=':', alpha=0.5)

    fig.suptitle('唯一主方法 C3 与 A4b 等全部基线同会话端到端公平测速（零跨协议延迟混比）', fontsize=13.4, fontweight='bold', y=0.96)
    out_p = FIG_DIR / 'fig2_c3_fair_end2end_latency_and_pareto.png'
    fig.savefig(out_p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_p}")


def plot_fig3_gray_and_blur_ring_mechanism(d2b: dict):
    mm = d2b['g1_to_g4_gray_and_blur_ring_metrics_3seed']
    bt = d2b['g1_to_g4_paired_bootstrap_and_did_1000']
    gm = d2b['outer_ring_125_geometry_stats']

    conds = [
        ('G1_125_RealBG_Iso30', 'G1: 1.25x 原始背景\n(单尺度参考)'),
        ('G2_125_GrayOuter_Iso30', 'G2: 1.25x 灰色外环\n(框外置 128 中性灰)'),
        ('G2_blur_125_BlurOuter_Iso30', 'G2_blur: 1.25x 模糊外环\n(框外 31x31 高斯模糊)'),
        ('G3_082_125_RealBG_C3_Locked', 'G3: 0.82x+1.25x 原背景\n(唯一冻结主方法 C3)'),
        ('G4_082_125_GrayOuter_C3_Locked', 'G4: 0.82x+1.25x 灰色外环\n(双尺度灰环干预)'),
        ('G4_blur_082_125_BlurOuter_C3_Locked', 'G4_blur: 0.82x+1.25x 模糊外环\n(双尺度模糊干预)'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.5, 6.9), gridspec_kw={'width_ratios': [1.10, 1.15]})
    fig.subplots_adjust(wspace=0.47, bottom=0.23, top=0.84)

    x = np.arange(len(conds))
    w = 0.36
    b_m = [mm[k]['testB']['AP50_95_mean'] for k, _ in conds]
    b_s = [mm[k]['testB']['AP50_95_std'] for k, _ in conds]
    c_m = [mm[k]['testC']['AP50_95_mean'] for k, _ in conds]
    c_s = [mm[k]['testC']['AP50_95_std'] for k, _ in conds]

    bars_b = ax1.bar(x - w / 2, b_m, w, yerr=b_s, capsize=3.2, color='#2563EB', alpha=0.88,
                     error_kw={'lw': 1.2, 'ecolor': '#1E3A8A'}, label='SciDB testB (自然田间多叶复杂背景)')
    bars_c = ax1.bar(x + w / 2, c_m, w, yerr=c_s, capsize=3.2, color='#059669', alpha=0.88,
                     error_kw={'lw': 1.2, 'ecolor': '#064E3B'}, label='SciDB testC (均匀白板摘叶背景)')

    bars_b[3].set_edgecolor('#B91C1C'); bars_b[3].set_linewidth(2.2)
    bars_c[3].set_edgecolor('#B91C1C'); bars_c[3].set_linewidth(2.2)

    for rect, val, sd in zip(bars_b, b_m, b_s):
        ax1.text(rect.get_x() + rect.get_width() / 2, val + sd + 0.35, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=8.0, color='#1E3A8A', fontweight='bold')
    for rect, val, sd in zip(bars_c, c_m, c_s):
        ax1.text(rect.get_x() + rect.get_width() / 2, val + sd + 0.35, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=8.0, color='#065F46', fontweight='bold')

    y_lo = min(min(b_m), min(c_m)) - 3.5
    y_hi = max(max(b_m) + max(b_s), max(c_m) + max(c_s)) + 2.8
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in conds], fontsize=8.4, rotation=22, ha='right')
    ax1.set_ylabel('COCO AP50-95 (%, 3随机种子均值±标准差)', fontsize=10.4)
    b_ofrac = gm['testB']['mean_outer_ring_area_pct_in_224']
    ax1.set_title(f'(a) G1–G4 同几何灰环与模糊外环实验 (1.25x 框外平均面积占比 {b_ofrac:.1f}%)', fontsize=10.8, fontweight='bold', pad=12)
    ax1.grid(axis='y', ls=':', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=9.0, frameon=True)

    forest_items = [
        ('G2_125_GrayOuter_Iso30__minus__G1_125_RealBG_Iso30', 'G2 - G1 (单尺度灰环直接效应)'),
        ('G2_blur_125_BlurOuter_Iso30__minus__G1_125_RealBG_Iso30', 'G2_blur - G1 (单尺度模糊外环直接效应)'),
        ('G3_082_125_RealBG_C3_Locked__minus__G1_125_RealBG_Iso30', 'Δ_real = G3 - G1 (原始背景双尺度增益)'),
        ('G4_082_125_GrayOuter_C3_Locked__minus__G2_125_GrayOuter_Iso30', 'Δ_gray = G4 - G2 (灰环后双尺度增益)'),
        ('G4_blur_082_125_BlurOuter_C3_Locked__minus__G2_blur_125_BlurOuter_Iso30', 'Δ_blur = G4_blur - G2_blur (模糊后双尺度增益)'),
        ('DiD_Gray_minus_Real_LockedC3', 'DiD_gray = Δ_gray - Δ_real (灰环双重差分)'),
        ('DiD_Blur_minus_Real_LockedC3', 'DiD_blur = Δ_blur - Δ_real (模糊双重差分)'),
    ]
    y_pos = np.arange(len(forest_items))[::-1]
    off = 0.17
    ax2.axvline(0.0, color='#DC2626', ls='--', lw=1.4, alpha=0.85, label='零效应线 (ΔAP = 0)')

    for yi, (pk, plbl) in zip(y_pos, forest_items):
        rb = bt['testB'][pk]['AP50_95']
        rc = bt['testC'][pk]['AP50_95']
        mb, (lb, hb), pb = rb['mean_diff_3seed'], rb['ci95_3seed'], rb['p_value_two_sided']
        mc, (lc, hc), pc = rc['mean_diff_3seed'], rc['ci95_3seed'], rc['p_value_two_sided']

        ax2.errorbar(mb, yi + off, xerr=[[mb - lb], [hb - mb]], fmt='o', color='#2563EB', capsize=3.8, lw=1.7, ms=6.0,
                     label='SciDB testB (95% CI)' if yi == y_pos[0] else None)
        ax2.errorbar(mc, yi - off, xerr=[[mc - lc], [hc - mc]], fmt='s', color='#059669', capsize=3.8, lw=1.7, ms=6.0,
                     label='SciDB testC (95% CI)' if yi == y_pos[0] else None)

        ax2.text(hb + 0.07, yi + off, f'{mb:+.2f}% (p={pb:.3f})', va='center', ha='left', fontsize=7.8, color='#1E40AF', fontweight='bold')
        ax2.text(hc + 0.07, yi - off, f'{mc:+.2f}% (p={pc:.3f})', va='center', ha='left', fontsize=7.8, color='#065F46', fontweight='bold')

    all_hi = [max(bt['testB'][pk]['AP50_95']['ci95_3seed'][1], bt['testC'][pk]['AP50_95']['ci95_3seed'][1]) for pk, _ in forest_items]
    all_lo = [min(bt['testB'][pk]['AP50_95']['ci95_3seed'][0], bt['testC'][pk]['AP50_95']['ci95_3seed'][0]) for pk, _ in forest_items]
    ax2.set_xlim(min(-1.5, min(all_lo) - 0.55), max(all_hi) + 2.45)
    ax2.set_ylim(-0.65, len(forest_items) - 0.20)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([lbl for _, lbl in forest_items], fontsize=8.8)
    ax2.set_xlabel('3种子同步逐图配对 Bootstrap ΔAP50-95 (%) 及 95% 置信区间', fontsize=10.2)
    ax2.set_title('(b) 背景干预直接效应、双尺度增益与双重差分 (DiD, B=1,000)', fontsize=10.8, fontweight='bold', pad=12)
    ax2.grid(axis='x', ls=':', alpha=0.5)
    ax2.legend(loc='lower right', fontsize=8.7, frameon=True)

    fig.suptitle('主方法 C3 受控同几何灰环与模糊外环背景干预机制实验 (G1–G4)', fontsize=13.4, fontweight='bold', y=0.96)
    out_p = FIG_DIR / 'fig3_g1_to_g4_gray_and_blur_ring_mechanism.png'
    fig.savefig(out_p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_p}")


def main():
    d2 = json.loads(STAGE2_JSON.read_text(encoding='utf-8'))
    d2b = json.loads(STAGE2B_JSON.read_text(encoding='utf-8'))
    plot_fig1_scale_ablation_and_bootstrap(d2)
    plot_fig2_fair_latency_profile(d2)
    plot_fig3_gray_and_blur_ring_mechanism(d2b)


if __name__ == '__main__':
    main()
