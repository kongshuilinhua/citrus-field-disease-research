# -*- coding: utf-8 -*-
"""
绘制严格等检测器权重 (w_det=0.30) 下 C3 的 A1–A5 受控尺度消融、配对 Bootstrap 森林图与端到端公平测速图
(generate_c3_ablation_figures.py)
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
RES_JSON = V5_DIR / 'results' / 'stage2_c3_controlled_scale_ablation_complete.json'
FIG_DIR = V5_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_fig1_scale_ablation_and_bootstrap(data: dict):
    ab = data['ablation_metrics_3seed']
    bt = data['paired_bootstrap_1000_results']

    items = [
        ('B0_Native_YOLOv8n', 'B0: YOLOv8n\n(单阶段基线)'),
        ('B1_Full_DINOv2_125_Hard', 'B1: Full DINOv2\n(1.25x 硬覆盖)'),
        ('A1_Single_125_Iso30', 'A1: 1.25x 单尺度\n(标准视角, w=0.3/0.7)'),
        ('A2_Single_082_Iso30', 'A2: 0.82x 单尺度\n(紧致视角, w=0.3/0.7)'),
        ('A4a_CloseDual_115_135_Iso30_ValGrid', 'A4a: 1.15x+1.35x\n(相近双裁剪TTA)'),
        ('A4b_FlipTTA_125_HFlip_Iso30_ValGrid', 'A4b: 1.25x+翻转\n(常规双视角TTA)'),
        ('A3_C3_Locked_082_125_Main', 'A3/C3: 0.82x+1.25x\n(互补双尺度主方法)'),
        ('A5_TriScale_082_125_250_Iso30_ValGrid', 'A5: +2.50x 三尺度\n(追加大范围环境)'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.5, 6.8), gridspec_kw={'width_ratios': [1.28, 1.02]})
    fig.subplots_adjust(wspace=0.30, bottom=0.22, top=0.85)

    x = np.arange(len(items))
    w = 0.37
    b_m = [ab[k]['testB']['AP50_95_mean'] for k, _ in items]
    b_s = [ab[k]['testB']['AP50_95_std'] for k, _ in items]
    c_m = [ab[k]['testC']['AP50_95_mean'] for k, _ in items]
    c_s = [ab[k]['testC']['AP50_95_std'] for k, _ in items]

    bars_b = ax1.bar(x - w / 2, b_m, w, yerr=b_s, capsize=3.2, color='#3B82F6', alpha=0.88,
                     error_kw={'lw': 1.2, 'ecolor': '#1E3A8A'}, label='SciDB testB (自然田间复杂背景)')
    bars_c = ax1.bar(x + w / 2, c_m, w, yerr=c_s, capsize=3.2, color='#10B981', alpha=0.88,
                     error_kw={'lw': 1.2, 'ecolor': '#064E3B'}, label='SciDB testC (高光白板异质背景)')

    bars_b[6].set_edgecolor('#B91C1C'); bars_b[6].set_linewidth(2.1)
    bars_c[6].set_edgecolor('#B91C1C'); bars_c[6].set_linewidth(2.1)

    for rect, val, sd in zip(bars_b, b_m, b_s):
        ax1.text(rect.get_x() + rect.get_width() / 2, val + sd + 0.45, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=7.9, color='#1E3A8A', fontweight='bold')
    for rect, val, sd in zip(bars_c, c_m, c_s):
        ax1.text(rect.get_x() + rect.get_width() / 2, val + sd + 0.45, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=7.9, color='#065F46', fontweight='bold')

    ax1.set_ylim(63, 93.0)
    ax1.set_ylabel('COCO AP50-95 (%, 3随机种子均值±标准差)', fontsize=10.5)
    ax1.set_title('(a) 严格固定 w_det=0.30 受控尺度消融 A1–A5 (同冻结骨干、零MLP头)', fontsize=11.1, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in items], fontsize=8.6, rotation=22, ha='right')
    ax1.grid(axis='y', ls=':', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=9.2, frameon=True)

    forest_pairs = [
        ('A3_C3_Locked_082_125_Main__minus__A1_Single_125_Iso30', 'A3(C3) - A1(1.25x 标准单尺度)'),
        ('A3_C3_Locked_082_125_Main__minus__A2_Single_082_Iso30', 'A3(C3) - A2(0.82x 紧致单尺度)'),
        ('A3_C3_Locked_082_125_Main__minus__A4a_CloseDual_115_135_Iso30_ValGrid', 'A3(C3) - A4a(1.15x+1.35x 相近TTA)'),
        ('A3_C3_Locked_082_125_Main__minus__A4b_FlipTTA_125_HFlip_Iso30_ValGrid', 'A3(C3) - A4b(1.25x+翻转 常规TTA)'),
        ('A5_TriScale_082_125_250_Iso30_ValGrid__minus__A3_C3_Locked_082_125_Main', 'A5(+2.50x环境视角) - A3(C3)'),
    ]
    y_pos = np.arange(len(forest_pairs))[::-1]
    off = 0.17

    ax2.axvline(0.0, color='#DC2626', ls='--', lw=1.4, alpha=0.85, label='零差异基准线 (ΔAP = 0)')
    for yi, (pk, plbl) in zip(y_pos, forest_pairs):
        rb = bt['testB'][pk]['AP50_95']
        rc = bt['testC'][pk]['AP50_95']
        mb, (lb, hb), pb = rb['mean_diff_3seed'], rb['ci95_3seed'], rb['p_value_two_sided']
        mc, (lc, hc), pc = rc['mean_diff_3seed'], rc['ci95_3seed'], rc['p_value_two_sided']

        ax2.errorbar(mb, yi + off, xerr=[[mb - lb], [hb - mb]], fmt='o', color='#2563EB', capsize=4, lw=1.8, ms=6.5,
                     label='SciDB testB (95% CI)' if yi == y_pos[0] else None)
        ax2.errorbar(mc, yi - off, xerr=[[mc - lc], [hc - mc]], fmt='s', color='#059669', capsize=4, lw=1.8, ms=6.5,
                     label='SciDB testC (95% CI)' if yi == y_pos[0] else None)

        ax2.text(hb + 0.08, yi + off, f'{mb:+.2f}% (p={pb:.3f})', va='center', ha='left', fontsize=8.1, color='#1E40AF', fontweight='bold')
        ax2.text(hc + 0.08, yi - off, f'{mc:+.2f}% (p={pc:.3f})', va='center', ha='left', fontsize=8.1, color='#065F46', fontweight='bold')

    all_hi = [max(bt['testB'][pk]['AP50_95']['ci95_3seed'][1], bt['testC'][pk]['AP50_95']['ci95_3seed'][1]) for pk, _ in forest_pairs]
    all_lo = [min(bt['testB'][pk]['AP50_95']['ci95_3seed'][0], bt['testC'][pk]['AP50_95']['ci95_3seed'][0]) for pk, _ in forest_pairs]
    ax2.set_xlim(min(-1.2, min(all_lo) - 0.5), max(all_hi) + 2.15)
    ax2.set_ylim(-0.65, len(forest_pairs) - 0.20)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([lbl for _, lbl in forest_pairs], fontsize=9.1)
    ax2.set_xlabel('3种子同步逐图配对 Bootstrap ΔAP50-95 (%) 及 95% 置信区间', fontsize=10.2)
    ax2.set_title('(b) 等检测器权重 (w_det=0.30) 关键假设配对检验森林图 (B=1,000)', fontsize=11.1, fontweight='bold', pad=12)
    ax2.grid(axis='x', ls=':', alpha=0.5)
    ax2.legend(loc='lower right', fontsize=8.8, frameon=True)

    fig.suptitle('主方法 C3 (0.82× + 1.25× 双尺度校准后验融合) 严格受控尺度消融与配对统计检验', fontsize=13.5, fontweight='bold', y=0.96)
    out_p = FIG_DIR / 'fig1_c3_controlled_scale_ablation_and_bootstrap.png'
    fig.savefig(out_p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_p}")


def plot_fig2_fair_latency_profile(data: dict):
    lat = data['fair_end2end_latency_rtx4060ti_fp32']
    ab = data['ablation_metrics_3seed']

    modes = [
        ('B0_Native_YOLOv8n', 'B0: YOLOv8n', 'B0_Native_YOLOv8n', '#94A3B8'),
        ('B1_Full_DINOv2_125_Hard', 'B1: Full DINOv2 (1.25x)', 'B1_Full_DINOv2_125_Hard', '#64748B'),
        ('A1_Single_125', 'A1: 1.25x 单尺度校准', 'A1_Single_125_Iso30', '#3B82F6'),
        ('A2_Single_082', 'A2: 0.82x 紧致单尺度', 'A2_Single_082_Iso30', '#0EA5E9'),
        ('A4a_CloseDual_115_135', 'A4a: 1.15x+1.35x 相近TTA', 'A4a_CloseDual_115_135_Iso30_ValGrid', '#F59E0B'),
        ('A3_Dual_082_125_C3_Main', 'A3/C3: 0.82x+1.25x 主方法', 'A3_C3_Locked_082_125_Main', '#10B981'),
        ('A5_TriScale_082_125_250', 'A5: 0.82x+1.25x+2.50x 三尺度', 'A5_TriScale_082_125_250_Iso30_ValGrid', '#8B5CF6'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.8, 6.5))
    fig.subplots_adjust(wspace=0.24, bottom=0.22, top=0.85)

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
            ax1.text(r.get_x() + r.get_width() / 2, v + 0.6, f'{v:.1f}', ha='center', va='bottom', fontsize=7.4, fontweight='bold', rotation=30)

    ax1.set_ylim(0, max(p95_val) * 1.25)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl, _, _ in modes], fontsize=8.8, rotation=22, ha='right')
    ax1.set_ylabel('端到端单图总推理耗时 (ms, RTX 4060 Ti FP32)', fontsize=10.4)
    ax1.set_title('(a) 含解码+检测+裁剪+分类+融合+NMS 的公平延迟剖面 (SciDB val)', fontsize=11.0, fontweight='bold', pad=11)
    ax1.grid(axis='y', ls=':', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=8.9, frameon=True)

    offsets = [
        (10, -12),   # B0
        (10, -22),   # B1
        (-12, -6),   # A1
        (-12, -16),  # A2
        (12, -8),    # A4a
        (-8, 14),    # A3/C3
        (10, 8),     # A5
    ]
    for (mk, mlbl, ab_k, col), (ox, oy) in zip(modes, offsets):
        ms_c = lat[mk]['testC_all602imgs']['mean_ms']
        ap_b = ab[ab_k]['testB']['AP50_95_mean']
        ap_c = ab[ab_k]['testC']['AP50_95_mean']
        avg_bc = 0.5 * (ap_b + ap_c)
        mk_sz = 145 if 'C3_Main' in mk else 90
        edge = '#B91C1C' if 'C3_Main' in mk else 'black'
        ax2.scatter(ms_c, avg_bc, s=mk_sz, color=col, edgecolors=edge, linewidths=1.9 if 'C3_Main' in mk else 1.0, zorder=5)
        ha = 'right' if ox < 0 else 'left'
        ax2.annotate(f'{mlbl}\n({ms_c:.1f}ms, 均值AP={avg_bc:.2f}%)',
                     (ms_c, avg_bc), textcoords='offset points', xytext=(ox, oy),
                     ha=ha, fontsize=8.2, fontweight='bold' if 'C3_Main' in mk else 'normal',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=col, alpha=0.88))

    ms_vals = [lat[mk]['testC_all602imgs']['mean_ms'] for mk, _, _, _ in modes]
    ap_vals = [0.5 * (ab[ab_k]['testB']['AP50_95_mean'] + ab[ab_k]['testC']['AP50_95_mean']) for _, _, ab_k, _ in modes]
    ax2.set_xlim(min(ms_vals) - 2.5, max(ms_vals) + 6.8)
    ax2.set_ylim(min(ap_vals) - 1.8, max(ap_vals) + 2.2)
    ax2.set_xlabel('端到端单图平均推理延迟 (ms, SciDB testC 602图实测)', fontsize=10.4)
    ax2.set_ylabel('SciDB testB & testC 平均 AP50-95 (%, 3种子均值)', fontsize=10.4)
    ax2.set_title('(b) 受控尺度消融精度—延迟帕累托权衡 (单批次并行裁剪)', fontsize=11.0, fontweight='bold', pad=11)
    ax2.grid(True, ls=':', alpha=0.5)

    fig.suptitle('主方法 C3 端到端公平测速（含多目标场景与 P95 尾部延迟）及精度—速度权衡', fontsize=13.5, fontweight='bold', y=0.96)
    out_p = FIG_DIR / 'fig2_c3_fair_end2end_latency_and_pareto.png'
    fig.savefig(out_p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_p}")


def main():
    data = json.loads(RES_JSON.read_text(encoding='utf-8'))
    plot_fig1_scale_ablation_and_bootstrap(data)
    plot_fig2_fair_latency_profile(data)


if __name__ == '__main__':
    main()
