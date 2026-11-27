# -*- coding: utf-8 -*-
"""
绘制 Stage 2b v1.1 全部无重叠论文级图表（全图叠加 3 随机种子独立散点，杜绝仅报均值掩盖种子差异）：
1. fig1_c3_controlled_scale_ablation_and_bootstrap.png (受控尺度消融突出零调参等权 C3_equal 主口径 + 冻结 C3 对照 & 配对 Bootstrap)
2. fig2_c3_fair_end2end_latency_and_pareto.png (含 G2_blur 单刀遮罩与 A4b 的 10 组端到端延迟剖面与精度—延迟权衡)
3. fig3_g1_to_g4_gray_and_blur_ring_mechanism.png (G1–G4 灰环与模糊外环逐种子散点柱状图 + 含 G3−G2 / G3−G2_blur 关键判决项的森林图)
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


def overlay_seed_dots(ax, x_centers, seeds_list, color_dot):
    offsets = [-0.07, 0.0, 0.07]
    for xc, svals in zip(x_centers, seeds_list):
        for dx, sv in zip(offsets, svals):
            ax.scatter(xc + dx, sv, s=16, color=color_dot, edgecolors='white', linewidths=0.45, zorder=6, alpha=0.9)


def plot_fig1_scale_ablation_and_bootstrap(data: dict):
    ab = data['ablation_metrics_3seed']
    bt = data['paired_bootstrap_1000_results']

    items = [
        ('B0_Native_YOLOv8n', 'B0: YOLOv8n\n(单阶段基线)'),
        ('B1_Full_DINOv2_125_Hard', 'B1: Full DINOv2\n(1.25x 硬覆盖)'),
        ('A1_Single_125_Iso30', 'A1: 1.25x 单尺度\n(标准视角 0.3/0.7)'),
        ('A2_Single_082_Iso30', 'A2: 0.82x 单尺度\n(紧致视角 0.3/0.7)'),
        ('A4a_CloseDual_115_135_Iso30_ValGrid', 'A4a: 1.15x+1.35x\n(相近双裁剪TTA)'),
        ('A4b_FlipTTA_125_HFlip_Iso30_ValGrid', 'A4b: 1.25x+翻转\n(常规双视角TTA)'),
        ('A3_Dual_082_125_Iso30_Equal', 'C3_equal: 0.82x+1.25x\n(主表零调参 0.35/0.35)'),
        ('A3_C3_Locked_082_125_Main', 'C3_frozen: 0.82x+1.25x\n(冻结权重 0.20/0.50)'),
        ('A5_TriScale_082_125_250_Iso30_ValGrid', 'A5: +2.50x 三尺度\n(追加大范围环境)'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.6, 7.1), gridspec_kw={'width_ratios': [1.30, 1.12]})
    fig.subplots_adjust(wspace=0.45, bottom=0.23, top=0.85)

    x = np.arange(len(items))
    w = 0.38
    b_m = [ab[k]['testB']['AP50_95_mean'] for k, _ in items]
    b_s = [ab[k]['testB']['AP50_95_std'] for k, _ in items]
    b_seeds = [ab[k]['testB']['AP50_95_seeds'] for k, _ in items]
    c_m = [ab[k]['testC']['AP50_95_mean'] for k, _ in items]
    c_s = [ab[k]['testC']['AP50_95_std'] for k, _ in items]
    c_seeds = [ab[k]['testC']['AP50_95_seeds'] for k, _ in items]

    bars_b = ax1.bar(x - w / 2, b_m, w, yerr=b_s, capsize=3.0, color='#3B82F6', alpha=0.86,
                     error_kw={'lw': 1.1, 'ecolor': '#1E3A8A'}, label='SciDB testB (均值±SD，圆点为3种子)')
    bars_c = ax1.bar(x + w / 2, c_m, w, yerr=c_s, capsize=3.0, color='#10B981', alpha=0.86,
                     error_kw={'lw': 1.1, 'ecolor': '#064E3B'}, label='SciDB testC (均值±SD，圆点为3种子)')

    overlay_seed_dots(ax1, x - w / 2, b_seeds, '#1D4ED8')
    overlay_seed_dots(ax1, x + w / 2, c_seeds, '#047857')

    for hi_idx, col_e in [(6, '#D97706'), (7, '#B91C1C')]:
        bars_b[hi_idx].set_edgecolor(col_e); bars_b[hi_idx].set_linewidth(2.1)
        bars_c[hi_idx].set_edgecolor(col_e); bars_c[hi_idx].set_linewidth(2.1)

    for rect, val, sd, sds in zip(bars_b, b_m, b_s, b_seeds):
        top_y = max(val + sd, max(sds))
        ax1.text(rect.get_x() + rect.get_width() / 2, top_y + 0.42, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=7.4, color='#1E3A8A', fontweight='bold')
    for rect, val, sd, sds in zip(bars_c, c_m, c_s, c_seeds):
        top_y = max(val + sd, max(sds))
        ax1.text(rect.get_x() + rect.get_width() / 2, top_y + 0.42, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=7.4, color='#065F46', fontweight='bold')

    ax1.set_ylim(63, 93.5)
    ax1.set_ylabel('COCO AP50-95 (%, 逐种子散点 + 3种子均值)', fontsize=10.4)
    ax1.set_title('(a) 固定 w_det=0.30 受控尺度消融 (主表零调参 C3_equal 与冻结 C3 对照)', fontsize=10.9, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in items], fontsize=8.2, rotation=24, ha='right')
    ax1.grid(axis='y', ls=':', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=8.9, frameon=True)

    forest_pairs = [
        ('A3_Dual_082_125_Iso30_Equal__minus__A1_Single_125_Iso30', 'C3_equal - A1 (1.25x 标准单尺度)'),
        ('A3_Dual_082_125_Iso30_Equal__minus__A2_Single_082_Iso30', 'C3_equal - A2 (0.82x 紧致单尺度)'),
        ('A3_Dual_082_125_Iso30_Equal__minus__A4a_CloseDual_115_135_Iso30_Equal', 'C3_equal - A4a_eq (相近双裁剪TTA)'),
        ('A3_Dual_082_125_Iso30_Equal__minus__A4b_FlipTTA_125_HFlip_Iso30_Equal', 'C3_equal - A4b_eq (水平翻转TTA)'),
        ('A3_C3_Locked_082_125_Main__minus__A1_Single_125_Iso30', 'C3_frozen - A1 (冻结版 vs 单尺度)'),
        ('A3_C3_Locked_082_125_Main__minus__A4b_FlipTTA_125_HFlip_Iso30_ValGrid', 'C3_frozen - A4b (冻结版 vs 翻转TTA)'),
        ('A5_TriScale_082_125_250_Iso30_Equal__minus__A3_Dual_082_125_Iso30_Equal', 'A5_eq (+2.50x 三尺度) - C3_equal'),
    ]
    y_pos = np.arange(len(forest_pairs))[::-1]
    off = 0.17

    ax2.axvline(0.0, color='#DC2626', ls='--', lw=1.4, alpha=0.85, label='零差异基准线 (ΔAP = 0)')
    for yi, (pk, plbl) in zip(y_pos, forest_pairs):
        rb = bt['testB'][pk]['AP50_95']
        rc = bt['testC'][pk]['AP50_95']
        mb, (lb, hb), pb = rb['mean_diff_3seed'], rb['ci95_3seed'], rb['p_value_two_sided']
        mc, (lc, hc), pc = rc['mean_diff_3seed'], rc['ci95_3seed'], rc['p_value_two_sided']

        ax2.errorbar(mb, yi + off, xerr=[[mb - lb], [hb - mb]], fmt='o', color='#2563EB', capsize=3.5, lw=1.6, ms=5.8,
                     label='SciDB testB (95% CI)' if yi == y_pos[0] else None)
        ax2.errorbar(mc, yi - off, xerr=[[mc - lc], [hc - mc]], fmt='s', color='#059669', capsize=3.5, lw=1.6, ms=5.8,
                     label='SciDB testC (95% CI)' if yi == y_pos[0] else None)

        ax2.text(hb + 0.08, yi + off, f'{mb:+.2f}% (p={pb:.3f})', va='center', ha='left', fontsize=7.6, color='#1E40AF', fontweight='bold')
        ax2.text(hc + 0.08, yi - off, f'{mc:+.2f}% (p={pc:.3f})', va='center', ha='left', fontsize=7.6, color='#065F46', fontweight='bold')

    all_hi = [max(bt['testB'][pk]['AP50_95']['ci95_3seed'][1], bt['testC'][pk]['AP50_95']['ci95_3seed'][1]) for pk, _ in forest_pairs]
    all_lo = [min(bt['testB'][pk]['AP50_95']['ci95_3seed'][0], bt['testC'][pk]['AP50_95']['ci95_3seed'][0]) for pk, _ in forest_pairs]
    ax2.set_xlim(min(-1.3, min(all_lo) - 0.5), max(all_hi) + 2.35)
    ax2.set_ylim(-0.65, len(forest_pairs) - 0.20)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([lbl for _, lbl in forest_pairs], fontsize=8.6)
    ax2.set_xlabel('配对 Bootstrap ΔAP50-95 (%) 及 95% 置信区间 (B=1,000)', fontsize=10.1)
    ax2.set_title('(b) 零调参等权 C3_equal 与冻结版关键配对检验森林图', fontsize=10.9, fontweight='bold', pad=12)
    ax2.grid(axis='x', ls=':', alpha=0.5)
    ax2.legend(loc='lower right', fontsize=8.6, frameon=True)

    fig.suptitle('受控双尺度消融 v1.1：零调参等权 C3_equal 主口径与冻结 C3 逐种子及配对检验', fontsize=13.3, fontweight='bold', y=0.96)
    out_p = FIG_DIR / 'fig1_c3_controlled_scale_ablation_and_bootstrap.png'
    fig.savefig(out_p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_p}")


def plot_fig2_fair_latency_profile(data: dict, d2b: dict):
    lat = data['fair_end2end_latency_rtx4060ti_fp32']
    ab = data['ablation_metrics_3seed']
    mm = d2b['g1_to_g4_gray_and_blur_ring_metrics_3seed']

    modes = [
        ('B0_Native_YOLOv8n', 'B0: YOLOv8n', ab['B0_Native_YOLOv8n'], '#94A3B8'),
        ('B1_Full_DINOv2_125_Hard', 'B1: Full DINOv2', ab['B1_Full_DINOv2_125_Hard'], '#64748B'),
        ('C1_Single_125_035_065', 'C1: 1.25x (0.35/0.65)', ab['C1_Historical_125_035_065'], '#2563EB'),
        ('A1_Single_125', 'A1: 1.25x (0.30/0.70)', ab['A1_Single_125_Iso30'], '#3B82F6'),
        ('G2_blur_Single_125_BlurRing', 'G2_blur: 1.25x+模糊环', mm['G2_blur_125_BlurOuter_Iso30'], '#14B8A6'),
        ('A2_Single_082', 'A2: 0.82x 紧致单尺度', ab['A2_Single_082_Iso30'], '#0EA5E9'),
        ('A4a_CloseDual_115_135', 'A4a: 1.15x+1.35x TTA', ab['A4a_CloseDual_115_135_Iso30_ValGrid'], '#F59E0B'),
        ('A4b_FlipTTA_125_HFlip', 'A4b: 1.25x+翻转 TTA', ab['A4b_FlipTTA_125_HFlip_Iso30_ValGrid'], '#EC4899'),
        ('A3_Dual_082_125_C3_Main', 'C3: 0.82x+1.25x (等权/冻结)', ab['A3_Dual_082_125_Iso30_Equal'], '#10B981'),
        ('A5_TriScale_082_125_250', 'A5: +2.50x 三尺度', ab['A5_TriScale_082_125_250_Iso30_ValGrid'], '#8B5CF6'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.2, 7.0), gridspec_kw={'width_ratios': [1.18, 1.08]})
    fig.subplots_adjust(wspace=0.25, bottom=0.24, top=0.85)

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
            ax1.text(r.get_x() + r.get_width() / 2, v + 0.55, f'{v:.1f}', ha='center', va='bottom', fontsize=6.4, fontweight='bold', rotation=38)

    ax1.set_ylim(0, max(p95_val) * 1.26)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl, _, _ in modes], fontsize=8.0, rotation=25, ha='right')
    ax1.set_ylabel('端到端单图总推理耗时 (ms, RTX 4060 Ti FP32)', fontsize=10.3)
    ax1.set_title('(a) 含 G2_blur 与 A4b 的 10 组端到端延迟剖面 (SciDB val)', fontsize=10.7, fontweight='bold', pad=11)
    ax1.grid(axis='y', ls=':', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=8.7, frameon=True)

    pts = []
    for mk, mlbl, m_dict, col in modes:
        ms_c = lat[mk]['testC_all602imgs']['mean_ms']
        ap_b = m_dict['testB']['AP50_95_mean']
        ap_c = m_dict['testC']['AP50_95_mean']
        extra = " [冻结79.19]" if 'C3_Main' in mk else ""
        pts.append((mk, mlbl + extra, ms_c, ap_b, ap_c, col))

    ms_vals = [p[2] for p in pts]
    apb_vals = [p[3] for p in pts]
    ax2.set_xlim(min(ms_vals) - 2.0, max(ms_vals) + 16.8)
    y_min, y_max = min(apb_vals) - 1.5, max(apb_vals) + 2.6
    ax2.set_ylim(y_min, y_max)

    pts_sorted = sorted(pts, key=lambda z: z[3])
    label_ys = np.linspace(y_min + 0.65, y_max - 0.55, len(pts_sorted))
    for (mk, mlbl, ms_c, ap_b, ap_c, col), ly in zip(pts_sorted, label_ys):
        is_hi = ('C3_Main' in mk) or ('G2_blur' in mk)
        mk_sz = 145 if is_hi else 84
        edge = '#B91C1C' if 'C3_Main' in mk else ('#0F766E' if 'G2_blur' in mk else 'black')
        ax2.scatter(ms_c, ap_b, s=mk_sz, color=col, edgecolors=edge, linewidths=2.0 if is_hi else 1.0, zorder=5)
        lx = max(ms_vals) + 0.7 if ms_c >= 14.0 else ms_c + 1.2
        ax2.annotate(
            f"{mlbl}: {ms_c:.2f}ms | B={ap_b:.2f}% / C={ap_c:.2f}%",
            xy=(ms_c, ap_b), xytext=(lx, ly),
            ha='left', va='center', fontsize=7.4,
            fontweight='bold' if is_hi else 'normal',
            arrowprops=dict(arrowstyle='->', color=col, lw=0.95, alpha=0.75),
            bbox=dict(boxstyle='round,pad=0.20', fc='#FEF2F2' if 'C3_Main' in mk else 'white', ec=edge if is_hi else col, alpha=0.92),
        )

    ax2.set_xlabel('端到端单图平均延迟 (ms, SciDB testC 602图实测)', fontsize=10.3)
    ax2.set_ylabel('复杂自然冠层 SciDB testB AP50-95 (%, 附 testC)', fontsize=10.3)
    ax2.set_title('(b) 复杂背景精度 vs 延迟权衡 (含单刀遮罩 G2_blur 与双刀 C3)', fontsize=10.7, fontweight='bold', pad=11)
    ax2.grid(True, ls=':', alpha=0.5)

    fig.suptitle('同硬件同协议 10 组方案端到端公平测速（含 G2_blur 单刀模糊遮罩与 A4b，零跨协议混比）', fontsize=13.2, fontweight='bold', y=0.96)
    out_p = FIG_DIR / 'fig2_c3_fair_end2end_latency_and_pareto.png'
    fig.savefig(out_p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_p}")


def plot_fig3_gray_and_blur_ring_mechanism(d2b: dict):
    mm = d2b['g1_to_g4_gray_and_blur_ring_metrics_3seed']
    bt = d2b['g1_to_g4_paired_bootstrap_and_did_1000']
    gm = d2b['outer_ring_125_geometry_stats']

    conds = [
        ('G1_125_RealBG_Iso30', 'G1: 1.25x 原背景\n(单尺度参考)'),
        ('G2_125_GrayOuter_Iso30', 'G2: 1.25x 灰色外环\n(单刀置灰)'),
        ('G2_blur_125_BlurOuter_Iso30', 'G2_blur: 1.25x 模糊环\n(单刀遮罩最简实现)'),
        ('G3_eq_082_125_RealBG_Equal', 'G3_eq: C3_equal\n(双刀等权 0.35/0.35)'),
        ('G3_082_125_RealBG_C3_Locked', 'G3: C3_frozen\n(双刀冻结 0.20/0.50)'),
        ('G4_082_125_GrayOuter_C3_Locked', 'G4: 双刀+灰环\n(冻结权重)'),
        ('G4_blur_082_125_BlurOuter_C3_Locked', 'G4_blur: 双刀+模糊环\n(冻结权重)'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.8, 7.1), gridspec_kw={'width_ratios': [1.14, 1.18]})
    fig.subplots_adjust(wspace=0.49, bottom=0.23, top=0.84)

    x = np.arange(len(conds))
    w = 0.37
    b_m = [mm[k]['testB']['AP50_95_mean'] for k, _ in conds]
    b_s = [mm[k]['testB']['AP50_95_std'] for k, _ in conds]
    b_seeds = [mm[k]['testB']['AP50_95_seeds'] for k, _ in conds]
    c_m = [mm[k]['testC']['AP50_95_mean'] for k, _ in conds]
    c_s = [mm[k]['testC']['AP50_95_std'] for k, _ in conds]
    c_seeds = [mm[k]['testC']['AP50_95_seeds'] for k, _ in conds]

    bars_b = ax1.bar(x - w / 2, b_m, w, yerr=b_s, capsize=3.0, color='#2563EB', alpha=0.86,
                     error_kw={'lw': 1.15, 'ecolor': '#1E3A8A'}, label='SciDB testB (均值±SD，圆点为3种子)')
    bars_c = ax1.bar(x + w / 2, c_m, w, yerr=c_s, capsize=3.0, color='#059669', alpha=0.86,
                     error_kw={'lw': 1.15, 'ecolor': '#064E3B'}, label='SciDB testC (均值±SD，圆点为3种子)')

    overlay_seed_dots(ax1, x - w / 2, b_seeds, '#1D4ED8')
    overlay_seed_dots(ax1, x + w / 2, c_seeds, '#047857')

    for idx_h, col_h in [(2, '#0F766E'), (3, '#D97706'), (4, '#B91C1C')]:
        bars_b[idx_h].set_edgecolor(col_h); bars_b[idx_h].set_linewidth(2.1)
        bars_c[idx_h].set_edgecolor(col_h); bars_c[idx_h].set_linewidth(2.1)

    for rect, val, sd, sds in zip(bars_b, b_m, b_s, b_seeds):
        top_y = max(val + sd, max(sds))
        ax1.text(rect.get_x() + rect.get_width() / 2, top_y + 0.32, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=7.6, color='#1E3A8A', fontweight='bold')
    for rect, val, sd, sds in zip(bars_c, c_m, c_s, c_seeds):
        top_y = max(val + sd, max(sds))
        ax1.text(rect.get_x() + rect.get_width() / 2, top_y + 0.32, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=7.6, color='#065F46', fontweight='bold')

    y_lo = min(min(min(s) for s in b_seeds), min(min(s) for s in c_seeds)) - 2.2
    y_hi = max(max(max(s) for s in b_seeds), max(max(s) for s in c_seeds)) + 2.6
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in conds], fontsize=8.1, rotation=23, ha='right')
    ax1.set_ylabel('COCO AP50-95 (%, 逐种子散点 + 3种子均值)', fontsize=10.3)
    b_ofrac = gm['testB']['mean_outer_ring_area_pct_in_224']
    ax1.set_title(f'(a) G1–G4 同几何干预逐种子与均值对比 (1.25x 框外平均占比 {b_ofrac:.1f}%)', fontsize=10.7, fontweight='bold', pad=12)
    ax1.grid(axis='y', ls=':', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=8.8, frameon=True)

    forest_items = [
        ('G2_blur_125_BlurOuter_Iso30__minus__G1_125_RealBG_Iso30', 'G2_blur - G1 (单刀模糊遮罩直接效应)'),
        ('G3_082_125_RealBG_C3_Locked__minus__G2_125_GrayOuter_Iso30', 'G3(冻结) - G2 (两刀 vs 单刀灰环)'),
        ('G3_082_125_RealBG_C3_Locked__minus__G2_blur_125_BlurOuter_Iso30', 'G3(冻结) - G2_blur (两刀 vs 单刀模糊环★)'),
        ('G3_eq_082_125_RealBG_Equal__minus__G2_blur_125_BlurOuter_Iso30', 'G3_eq(等权) - G2_blur (两刀等权 vs 单刀模糊环★)'),
        ('DiD_Gray_minus_Real_LockedC3', 'DiD_gray (冻结C3, 2/3种子负向, S0=+0.51)'),
        ('DiD_Blur_minus_Real_LockedC3', 'DiD_blur (冻结C3, 2/3种子负向, S0=-0.15)'),
        ('DiD_Gray_minus_Real_EqualWeight', 'DiD_gray_eq (等权C3, p=0.122 不显著)'),
        ('DiD_Blur_minus_Real_EqualWeight', 'DiD_blur_eq (等权C3, p=0.098 不显著)'),
    ]
    y_pos = np.arange(len(forest_items))[::-1]
    off = 0.17
    ax2.axvline(0.0, color='#DC2626', ls='--', lw=1.4, alpha=0.85, label='零效应线 (ΔAP = 0)')

    for yi, (pk, plbl) in zip(y_pos, forest_items):
        rb = bt['testB'][pk]['AP50_95']
        rc = bt['testC'][pk]['AP50_95']
        mb, (lb, hb), pb = rb['mean_diff_3seed'], rb['ci95_3seed'], rb['p_value_two_sided']
        mc, (lc, hc), pc = rc['mean_diff_3seed'], rc['ci95_3seed'], rc['p_value_two_sided']

        ax2.errorbar(mb, yi + off, xerr=[[mb - lb], [hb - mb]], fmt='o', color='#2563EB', capsize=3.5, lw=1.6, ms=5.6,
                     label='SciDB testB (95% CI)' if yi == y_pos[0] else None)
        ax2.errorbar(mc, yi - off, xerr=[[mc - lc], [hc - mc]], fmt='s', color='#059669', capsize=3.5, lw=1.6, ms=5.6,
                     label='SciDB testC (95% CI)' if yi == y_pos[0] else None)

        ax2.text(hb + 0.07, yi + off, f'{mb:+.2f}% (p={pb:.3f})', va='center', ha='left', fontsize=7.5, color='#1E40AF', fontweight='bold')
        ax2.text(hc + 0.07, yi - off, f'{mc:+.2f}% (p={pc:.3f})', va='center', ha='left', fontsize=7.5, color='#065F46', fontweight='bold')

    all_hi = [max(bt['testB'][pk]['AP50_95']['ci95_3seed'][1], bt['testC'][pk]['AP50_95']['ci95_3seed'][1]) for pk, _ in forest_items]
    all_lo = [min(bt['testB'][pk]['AP50_95']['ci95_3seed'][0], bt['testC'][pk]['AP50_95']['ci95_3seed'][0]) for pk, _ in forest_items]
    ax2.set_xlim(min(-1.8, min(all_lo) - 0.45), max(all_hi) + 2.45)
    ax2.set_ylim(-0.65, len(forest_items) - 0.20)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([lbl for _, lbl in forest_items], fontsize=8.4)
    ax2.set_xlabel('配对 Bootstrap ΔAP50-95 (%) 及 95% 置信区间 (B=1,000)', fontsize=10.1)
    ax2.set_title('(b) “两刀 vs 单刀遮罩”判决检验与逐种子诚实勘误森林图', fontsize=10.7, fontweight='bold', pad=12)
    ax2.grid(axis='x', ls=':', alpha=0.5)
    ax2.legend(loc='lower right', fontsize=8.6, frameon=True)

    fig.suptitle('G1–G4 同几何背景干预机制实验 v1.1：单刀模糊遮罩 vs 双尺度等效增强检验', fontsize=13.2, fontweight='bold', y=0.96)
    out_p = FIG_DIR / 'fig3_g1_to_g4_gray_and_blur_ring_mechanism.png'
    fig.savefig(out_p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_p}")


def main():
    d2 = json.loads(STAGE2_JSON.read_text(encoding='utf-8'))
    d2b = json.loads(STAGE2B_JSON.read_text(encoding='utf-8'))
    plot_fig1_scale_ablation_and_bootstrap(d2)
    plot_fig2_fair_latency_profile(d2, d2b)
    plot_fig3_gray_and_blur_ring_mechanism(d2b)


if __name__ == '__main__':
    main()
