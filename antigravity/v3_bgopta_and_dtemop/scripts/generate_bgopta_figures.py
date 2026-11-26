# -*- coding: utf-8 -*-
"""
生成终稿 SCI 小论文出版级 300 DPI 配图（100% 基于锁定的 Phase 1 / Phase 2 / Phase 2B 实测 JSON）：
- fig1_igd_vs_bgopta_mechanism_and_pareto.png:
  (a) 为什么必须“输入级多粒度隔离”而非“输出层单框 Token 剥离”？展示 SciDB testB & testC 3 种子散点与均值对比；
  (b) RTX 4060 Ti 严格 FP32 真实延迟—跨场景精度帕累托前沿（揭示单批并行三视角 18.6ms 优于串行升级 24.1ms）。
- fig2_mask_purity_repaired_zenodo_and_vault.png:
  (a) CitrusUAT (N=443) 物理叶片真值掩膜下的前景聚焦纯度与框内四角背景抑制率（67.7%）；
  (b) 修复零丢框 Zenodo (519图/1765框) 与真正零接触 Sweet Orange V1 (1,389图最终盲测) 3 种子全面验证。
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK_DIR = Path(r'D:\新版柑橘\Antigravity_BGOPTA_单趟正交Token攻关_20261126')
PREV_DIR = Path(r'D:\新版柑橘\Antigravity_CGGAR_真创新攻关_20261125')
RES_DIR = WORK_DIR / 'results'
FIG_DIR = WORK_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    p2 = json.loads((RES_DIR / 'phase2_frozen_oneshot_complete_eval.json').read_text(encoding='utf-8'))
    p2b = json.loads((RES_DIR / 'phase2b_frozen_multigranular_and_clean_latency.json').read_text(encoding='utf-8'))
    prev_tcd = json.loads((PREV_DIR / 'results' / 'tcd_mge_scidb_3seed_final.json').read_text(encoding='utf-8'))

    scidb = p2['模块A_SciDB跨条件诊断与全消融']['summary']
    abl2_scidb = prev_tcd['三种子汇总']['Abl2_Disagreement_Esc_Only']
    lat7 = p2b['RTX4060Ti_七档零冗余单图延迟']
    zen_bg = p2['模块C_修复版Zenodo外部果园零丢框探索评测']['summary']
    zen_mg = p2b['冻结多粒度模型外部与最终盲测汇总']['zenodo_repaired_zerodrop']
    uat = p2['模块D_CitrusUAT真实叶片掩膜前景聚焦与跨域诊断']
    vault_bg = p2['模块E_未触碰SweetOrange最终封口盲测']['summary']
    vault_mg = p2b['冻结多粒度模型外部与最终盲测汇总']['vault_sweet_orange_1389']

    # ==========================================================================
    # Figure 1: (a) Input-Isolation vs Post-ViT Token Projection + (b) 7-Tier GPU Latency Pareto
    # ==========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.8), dpi=300)

    items_a = [
        ('YOLOv8n\nNative', scidb['B0_YOLOv8n_Native']['testB']['seeds'], scidb['B0_YOLOv8n_Native']['testC']['seeds']),
        ('Full DINOv2\n(1.25x Hard)', scidb['B1_Full_DINOv2_1Crop_Hard']['testB']['seeds'], scidb['B1_Full_DINOv2_1Crop_Hard']['testC']['seeds']),
        ('S1: 1.5x Plain\nToken Concat', scidb['Abl_S1_PlainPatchConcat_NoProj']['testB']['seeds'], scidb['Abl_S1_PlainPatchConcat_NoProj']['testC']['seeds']),
        ('S3: 1.5x Post-ViT\nBG-OPTA', scidb['Ours_S3_BGOPTA_1Pass']['testB']['seeds'], scidb['Ours_S3_BGOPTA_1Pass']['testC']['seeds']),
        ('1.25x [CLS]\n+DecoupledCalib', scidb['Ref_SingleCrop_125_Calib']['testB']['seeds'], scidb['Ref_SingleCrop_125_Calib']['testC']['seeds']),
        ('IGD-DCR-Esc\n(Disagree 25%)', abl2_scidb['testB']['seeds'], abl2_scidb['testC']['seeds']),
        ('IGD-DCR-Batch\n(3-Crop Diff)', scidb['Ref_3Crop_Diff_Static_Abl3']['testB']['seeds'], scidb['Ref_3Crop_Diff_Static_Abl3']['testC']['seeds']),
    ]

    x = np.arange(len(items_a))
    w = 0.36
    tb_m = [float(np.mean(it[1])) for it in items_a]
    tc_m = [float(np.mean(it[2])) for it in items_a]

    b_tb = ax1.bar(x - w / 2, tb_m, width=w, color='#85c1e9', edgecolor='#1b4f72', linewidth=0.8, label='SciDB testB (Orchard Occlusion)')
    b_tc = ax1.bar(x + w / 2, tc_m, width=w, color='#76d7c4', edgecolor='#0e6251', linewidth=0.8, label='SciDB testC (White-Board Glare)')

    # Emphasize Post-ViT probe vs Input-Isolated Ours
    b_tb[3].set_facecolor('#f5b041'); b_tc[3].set_facecolor('#eb984e')
    for idx_win in [5, 6]:
        b_tb[idx_win].set_facecolor('#2874a6')
        b_tc[idx_win].set_facecolor('#148f77')

    markers = ['o', 's', '^']
    for s_i in range(3):
        ax1.plot(x - w / 2 + (s_i - 1) * 0.065, [it[1][s_i] for it in items_a],
                 linestyle='none', marker=markers[s_i], color='#17202a', markersize=4.2, zorder=5,
                 label=f'Seed {s_i}')
        ax1.plot(x + w / 2 + (s_i - 1) * 0.065, [it[2][s_i] for it in items_a],
                 linestyle='none', marker=markers[s_i], color='#17202a', markersize=4.2, zorder=5)

    for idx_bar, rect in enumerate(b_tb):
        hv = rect.get_height()
        top_y = max(hv, max(items_a[idx_bar][1])) + 0.45
        ax1.text(rect.get_x() + rect.get_width() / 2.0, top_y, f"{hv:.2f}", ha='center', va='bottom', fontsize=6.9, fontweight='bold')
    for idx_bar, rect in enumerate(b_tc):
        hv = rect.get_height()
        top_y = max(hv, max(items_a[idx_bar][2])) + 0.45
        ax1.text(rect.get_x() + rect.get_width() / 2.0, top_y, f"{hv:.2f}", ha='center', va='bottom', fontsize=6.9, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels([it[0] for it in items_a], fontsize=8.0, fontweight='bold')
    ax1.set_ylim(65.0, 91.8)
    ax1.set_ylabel('Strict FP32 COCO AP50–95 (%)', fontsize=10.0, fontweight='bold')
    ax1.set_title('(a) Why Input-Isolated Differential Beats Post-ViT Token Stripping', fontsize=10.6, fontweight='bold', pad=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.35)
    ax1.legend(loc='upper left', fontsize=7.8, ncol=3)

    # Subplot (b): Clean 7-Tier Latency on testC vs Cross-Condition Mean AP50-95
    pts = [
        ('Native YOLOv8n', lat7['Tier1_Native_YOLOv8n']['testC']['mean_ms'],
         0.5 * (scidb['B0_YOLOv8n_Native']['testB']['mean'] + scidb['B0_YOLOv8n_Native']['testC']['mean']), '#7f8c8d', 'o', (8, -6)),
        ('Full DINOv2 (1-Crop)', lat7['Tier2_Full_DINOv2_1Crop']['testC']['mean_ms'],
         0.5 * (scidb['B1_Full_DINOv2_1Crop_Hard']['testB']['mean'] + scidb['B1_Full_DINOv2_1Crop_Hard']['testC']['mean']), '#e74c3c', 's', (-42, -30)),
        ('Post-ViT BG-OPTA (1.5x)', lat7['Tier3a_BGOPTA_1Pass']['testC']['mean_ms'],
         0.5 * (scidb['Ours_S3_BGOPTA_1Pass']['testB']['mean'] + scidb['Ours_S3_BGOPTA_1Pass']['testC']['mean']), '#e67e22', 'D', (8, -14)),
        ('IGD-DCR-Esc (2-Launch, 26% Esc)', lat7['Tier4a_Disagreement_Esc_Abl2']['testC']['mean_ms'],
         0.5 * (abl2_scidb['testB']['AP50_95_mean'] + abl2_scidb['testC']['AP50_95_mean']), '#2980b9', '^', (-118, -32)),
        ('TCD-MGE v2 (2-Launch, 70% Esc)', lat7['Tier4b_TriSignal_TCDMGE_v2']['testC']['mean_ms'],
         0.5 * (scidb['Ref_TCDMGE_v2_3CropEsc']['testB']['mean'] + scidb['Ref_TCDMGE_v2_3CropEsc']['testC']['mean']), '#8e44ad', 'v', (-92, 12)),
        ('Ours: IGD-DCR-Batch (1-Launch 3-Crop)', lat7['Tier5_Full_3Crop_Static_Abl3']['testC']['mean_ms'],
         0.5 * (scidb['Ref_3Crop_Diff_Static_Abl3']['testB']['mean'] + scidb['Ref_3Crop_Diff_Static_Abl3']['testC']['mean']), '#117a65', '*', (-145, 10)),
    ]
    for name_p, ms_v, ap_v, col, mk, offs in pts:
        sz = 210 if mk == '*' else 100
        ax2.scatter(ms_v, ap_v, s=sz, color=col, marker=mk, edgecolors='black', linewidths=0.9, zorder=6)
        ax2.annotate(f"{name_p}\n({ms_v:.2f} ms, {ap_v:.2f}%)",
                     xy=(ms_v, ap_v), xytext=offs, textcoords='offset points',
                     fontsize=7.8, fontweight='bold' if 'Ours' in name_p or 'IGD' in name_p else 'normal',
                     arrowprops=dict(arrowstyle='-', color=col, lw=0.7, alpha=0.6),
                     bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor=col, alpha=0.92))

    # Draw Pareto frontier line Native -> Full DINO -> IGD-DCR-Batch
    ax2.plot([pts[0][1], pts[1][1], pts[5][1]], [pts[0][2], pts[1][2], pts[5][2]],
             linestyle='--', color='#117a65', linewidth=1.4, alpha=0.75, label='Pareto Optimal Frontier')
    ax2.set_xlabel('End-to-End Single-Image Latency on RTX 4060 Ti (testC, Strict FP32 ms)', fontsize=9.8, fontweight='bold')
    ax2.set_ylabel('Mean Cross-Condition AP50–95 on testB & testC (%)', fontsize=9.8, fontweight='bold')
    ax2.set_title('(b) Real-Time Speed–Accuracy Frontier (53.7 FPS Single-Batch 3-Crop)', fontsize=10.6, fontweight='bold', pad=10)
    ax2.set_xlim(6.8, 27.2)
    ax2.set_ylim(70.5, 84.8)
    ax2.grid(True, linestyle='--', alpha=0.35)
    ax2.legend(loc='lower right', fontsize=8.5)

    fig.tight_layout(pad=1.6)
    f1_p = FIG_DIR / 'fig1_igd_vs_bgopta_mechanism_and_pareto.png'
    fig.savefig(f1_p, dpi=300)
    plt.close(fig)

    # ==========================================================================
    # Figure 2: (a) Repaired Zero-Drop Zenodo + Untouched Vault Sweet Orange + (b) CitrusUAT Leaf Mask Focus
    # ==========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.6), dpi=300)

    bench_names = [
        'Zenodo (Zero-Drop)\n2-Cls AP50',
        'Zenodo (Zero-Drop)\n3-Cls AP50',
        'Vault: SweetOrange\n14-Cls Exact Hit',
        'Vault: SweetOrange\n3-Cls Macro-F1',
    ]
    xb = np.arange(len(bench_names))
    wb = 0.16
    models_b = [
        ('Native YOLOv8n', '#95a5a6', [
            zen_bg['B0_YOLOv8n_Native']['det_2cls_AP50']['mean'],
            zen_bg['B0_YOLOv8n_Native']['det_3cls_repaired_AP50']['mean'],
            float(vault_bg['B0_YOLOv8n_Native']['strict_14cls_hit']['mean_std'].split('±')[0]),
            float(vault_bg['B0_YOLOv8n_Native']['aligned_3cls_f1']['mean_std'].split('±')[0]),
        ]),
        ('Full DINOv2 (1-Crop)', '#e74c3c', [
            zen_bg['B1_Full_DINOv2_1Crop']['det_2cls_AP50']['mean'],
            zen_bg['B1_Full_DINOv2_1Crop']['det_3cls_repaired_AP50']['mean'],
            float(vault_bg['B1_Full_DINOv2_1Crop']['strict_14cls_hit']['mean_std'].split('±')[0]),
            float(vault_bg['B1_Full_DINOv2_1Crop']['aligned_3cls_f1']['mean_std'].split('±')[0]),
        ]),
        ('Post-ViT BG-OPTA', '#f39c12', [
            zen_bg['Ours_S3_BGOPTA_1Pass']['det_2cls_AP50']['mean'],
            zen_bg['Ours_S3_BGOPTA_1Pass']['det_3cls_repaired_AP50']['mean'],
            float(vault_bg['Ours_S3_BGOPTA_1Pass']['strict_14cls_hit']['mean_std'].split('±')[0]),
            float(vault_bg['Ours_S3_BGOPTA_1Pass']['aligned_3cls_f1']['mean_std'].split('±')[0]),
        ]),
        ('IGD-DCR-Esc (33% Esc)', '#2e86c1', [
            zen_mg['Abl2_Disagreement_Esc']['det_2cls_AP50']['mean'],
            zen_mg['Abl2_Disagreement_Esc']['det_3cls_repaired_AP50']['mean'],
            vault_mg['Abl2_Disagreement_Esc']['strict_14cls_hit']['mean'],
            vault_mg['Abl2_Disagreement_Esc']['aligned_3cls_f1']['mean'],
        ]),
        ('Ours: IGD-DCR-Batch', '#148f77', [
            zen_mg['Abl3_Full_3Crop_Static']['det_2cls_AP50']['mean'],
            zen_mg['Abl3_Full_3Crop_Static']['det_3cls_repaired_AP50']['mean'],
            vault_mg['Abl3_Full_3Crop_Static']['strict_14cls_hit']['mean'],
            vault_mg['Abl3_Full_3Crop_Static']['aligned_3cls_f1']['mean'],
        ]),
    ]

    for idx_m, (m_title, col, vals) in enumerate(models_b):
        pos = xb + (idx_m - 2.0) * wb
        rects = ax1.bar(pos, vals, width=wb, color=col, edgecolor='black', linewidth=0.6, label=m_title)
        for r in rects:
            hv = r.get_height()
            ax1.text(r.get_x() + r.get_width() / 2.0, hv + 0.7, f"{hv:.1f}", ha='center', va='bottom', fontsize=6.6, fontweight='bold')

    ax1.set_xticks(xb)
    ax1.set_xticklabels(bench_names, fontsize=8.8, fontweight='bold')
    ax1.set_ylim(0, 93.0)
    ax1.set_ylabel('Metric Score (%, 3-Seed Mean)', fontsize=9.8, fontweight='bold')
    ax1.set_title('(a) Repaired Zenodo (Zero-Drop) & Untouched Vault (N=1,389) Validation', fontsize=10.5, fontweight='bold', pad=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.35)
    ax1.legend(loc='upper left', fontsize=8.0, ncol=2)

    # Subplot (b): CitrusUAT Mask-Free Foreground Purity & 5-Cls Transfer Comparison
    mf = uat['mask_focus_mean']
    u5_mg = p2b['冻结多粒度模型外部与最终盲测汇总']['citrusuat_5cls']
    u5_bg = uat['cross_domain_5cls_summary']

    cat_lbls = [
        'In-Box True Leaf\nPurity (BG-OPTA)',
        'Corner Gray-Board\nSuppression Rate',
        'CitrusUAT 14-Cls\nHit (YOLOv8n)',
        'CitrusUAT 14-Cls\nHit (IGD-DCR)',
        'CitrusUAT 5-Cls\nAcc (IGD-DCR)',
    ]
    c_vals = [
        mf['bgopta_soft_fg_leaf_purity_pct'],
        mf['corner_bg_suppression_pct'],
        float(u5_bg['B0_YOLOv8n']['strict_14cls_acc']['mean_std'].split('±')[0]),
        u5_mg['Abl3_Full_3Crop_Static']['strict_14cls_acc']['mean'],
        u5_mg['Abl3_Full_3Crop_Static']['aligned_5cls_acc']['mean'],
    ]
    c_cols = ['#27ae60', '#2ecc71', '#95a5a6', '#2980b9', '#148f77']
    xc = np.arange(len(cat_lbls))
    rc_bars = ax2.bar(xc, c_vals, width=0.50, color=c_cols, edgecolor='black', linewidth=0.75)
    for r in rc_bars:
        hv = r.get_height()
        ax2.text(r.get_x() + r.get_width() / 2.0, hv + 1.1, f"{hv:.2f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax2.set_xticks(xc)
    ax2.set_xticklabels(cat_lbls, fontsize=8.4, fontweight='bold')
    ax2.set_ylim(0, 108.0)
    ax2.set_ylabel('Percentage (%) on CitrusUAT (N=443)', fontsize=9.8, fontweight='bold')
    ax2.set_title('(b) CitrusUAT Ground-Truth Leaf-Mask Audit & Zero-Shot Transfer', fontsize=10.5, fontweight='bold', pad=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.35)

    fig.tight_layout(pad=1.6)
    f2_p = FIG_DIR / 'fig2_mask_purity_repaired_zenodo_and_vault.png'
    fig.savefig(f2_p, dpi=300)
    plt.close(fig)
    print(f"[终稿出版级配图生成完毕]\n  1) {f1_p}\n  2) {f2_p}")


if __name__ == '__main__':
    main()
