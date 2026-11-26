# -*- coding: utf-8 -*-
"""
生成零包装、包含全部强基线与负结果的独立审计配图 (300 DPI)：
严格遵守：
1. 不使用任何新造方法名包装旧消融（严格沿用 B0 / B1a / B1b / B2(MGS-v1) / B3 / BG-OPTA(S3) / Abl1 / Pure-3Crop-TTA / Plain-3Crop-Concat / Abl2 / TCD-MGE(v2) / Abl3 原名）；
2. 完整展示负结果与强基线反超现象：
   - SciDB 上 MGS-v1 (B2) 与 BG-OPTA (1.50x) 的跨条件掉点，以及 Abl3 相对三尺度普通拼接的无统计显著性 (p=0.70 / p=0.43)；
   - 同几何背景干预中“有效目标分辨率下降 (-2.08/-2.02pt)”与“背景干扰 (-0.68/-2.56pt)”的真实比例，以及 BG-OPTA 在均匀灰/白底上的负向过剥离；
   - 含完整融合、重算分与 batched_nms 的 val / testC 端到端真实延迟（披露多框图 P95=46.7ms 及无任何多视角方法快于单框 DINOv2 的事实）；
   - CitrusUAT 上 Full DINOv2 (73.44% F1) 反超所有融合方法 (Abl3 69.88%, BG-OPTA 64.36%) 及“灰底色差启发式伪掩膜”属性标注；
   - Sweet Orange V1 (曾用于 v1 分类验证) 上子集 3类 F1 原生 YOLOv8n (80.24%) 反超二阶段方法 (Abl3 73.22%, Full DINOv2 61.27%) 的双指标完整并列。
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
    aud = json.loads((RES_DIR / 'rigorous_audit_suite_complete.json').read_text(encoding='utf-8'))
    prev_tcd = json.loads((PREV_DIR / 'results' / 'tcd_mge_scidb_3seed_final.json').read_text(encoding='utf-8'))

    tcd_agg = prev_tcd['三种子汇总']
    scidb_p2 = p2['模块A_SciDB跨条件诊断与全消融']['summary']
    pt_a2 = aud['审计2_三尺度普通拼接与差分逐种子配对消融及Bootstrap置信区间']['point_estimates']
    boot = aud['审计2_三尺度普通拼接与差分逐种子配对消融及Bootstrap置信区间']['paired_bootstrap']
    a1 = aud['审计1_同几何背景干预与分辨率分离实验']
    a4 = aud['审计4_含完整融合重算分与NMS的端到端公平延迟及实时AP核验']

    # ==========================================================================
    # Figure 1: (a) Complete SciDB Baselines + Negatives + Paired Bootstrap Note
    #           (b) Same-Geometry Background Intervention (Resolution vs BG)
    # ==========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 6.0), dpi=300)

    items = [
        ('B0: Native\nYOLOv8n', tcd_agg['B0_YOLOv8n_Native']['testB']['seeds'], tcd_agg['B0_YOLOv8n_Native']['testC']['seeds']),
        ('B2: MGS-v1\n(Neg Result)', tcd_agg['B2_MGS_DINOv2_v1']['testB']['seeds'], tcd_agg['B2_MGS_DINOv2_v1']['testC']['seeds']),
        ('BG-OPTA 1.5x\n(Neg on testC)', scidb_p2['Ours_S3_BGOPTA_1Pass']['testB']['seeds'], scidb_p2['Ours_S3_BGOPTA_1Pass']['testC']['seeds']),
        ('B1a: Full DINO\n(1.25x Gray)', tcd_agg['B1a_Full_DINOv2_GrayPad_Orig']['testB']['seeds'], tcd_agg['B1a_Full_DINOv2_GrayPad_Orig']['testC']['seeds']),
        ('B3: Static\nPost. Fuse', tcd_agg['B3_Static_Posterior_Fuse']['testB']['seeds'], tcd_agg['B3_Static_Posterior_Fuse']['testC']['seeds']),
        ('Abl1: 1.25x\n14Cls+Rew+NMS', pt_a2['testB']['Abl1_SingleCrop_14Cls_Calib'], pt_a2['testC']['Abl1_SingleCrop_14Cls_Calib']),
        ('Pure 3-Crop\nTTA (No MLP)', pt_a2['testB']['Pure_3Crop_TTA_NoMLP'], pt_a2['testC']['Pure_3Crop_TTA_NoMLP']),
        ('3-Crop Plain\nConcat MLP', pt_a2['testB']['Fused_PlainConcat_3Vec'], pt_a2['testC']['Fused_PlainConcat_3Vec']),
        ('Abl2: Disagree\nEscalation', tcd_agg['Abl2_Disagreement_Esc_Only']['testB']['seeds'], tcd_agg['Abl2_Disagreement_Esc_Only']['testC']['seeds']),
        ('TCD-MGE\n(v2 Tri-Sig)', tcd_agg['Ours_TCDMGE_Selective']['testB']['seeds'], tcd_agg['Ours_TCDMGE_Selective']['testC']['seeds']),
        ('Abl3: 3-Crop\nDiff Static', pt_a2['testB']['Abl3_Diff_StaticWeight'], pt_a2['testC']['Abl3_Diff_StaticWeight']),
    ]

    x = np.arange(len(items))
    w = 0.38
    tb_m = [float(np.mean(it[1])) for it in items]
    tc_m = [float(np.mean(it[2])) for it in items]

    b_tb = ax1.bar(x - w / 2, tb_m, width=w, color='#85c1e9', edgecolor='#1b4f72', linewidth=0.7, label='SciDB testB (Exploratory Orchard)')
    b_tc = ax1.bar(x + w / 2, tc_m, width=w, color='#76d7c4', edgecolor='#0e6251', linewidth=0.7, label='SciDB testC (Exploratory White-Board)')

    # Color negative probes distinctly in coral/amber
    for neg_i in [1, 2]:
        b_tb[neg_i].set_facecolor('#f5b041')
        b_tc[neg_i].set_facecolor('#e67e22')

    mk_list = ['o', 's', '^']
    for s_i in range(3):
        ax1.plot(x - w / 2 + (s_i - 1) * 0.06, [it[1][s_i] for it in items],
                 linestyle='none', marker=mk_list[s_i], color='#17202a', markersize=3.6, zorder=5,
                 label=f'Seed {s_i}')
        ax1.plot(x + w / 2 + (s_i - 1) * 0.06, [it[2][s_i] for it in items],
                 linestyle='none', marker=mk_list[s_i], color='#17202a', markersize=3.6, zorder=5)

    for i_b, rect in enumerate(b_tb):
        hv = rect.get_height()
        ty = max(hv, max(items[i_b][1])) + 0.40
        ax1.text(rect.get_x() + rect.get_width() / 2.0, ty, f"{hv:.2f}", ha='center', va='bottom', fontsize=6.0, fontweight='bold', rotation=45)
    for i_b, rect in enumerate(b_tc):
        hv = rect.get_height()
        ty = max(hv, max(items[i_b][2])) + 0.40
        ax1.text(rect.get_x() + rect.get_width() / 2.0, ty, f"{hv:.2f}", ha='center', va='bottom', fontsize=6.0, fontweight='bold', rotation=45)

    bt_b = boot['testB']['Abl3_Diff_StaticWeight__minus__Fused_PlainConcat_3Vec']['mean3seed_AP50_95']
    bt_c = boot['testC']['Abl3_Diff_StaticWeight__minus__Fused_PlainConcat_3Vec']['mean3seed_AP50_95']
    note_txt = (
        f"Paired Bootstrap (B=1000, Abl3 Diff vs 3-Crop Plain Concat):\n"
        f"• testB: ΔAP = +{bt_b['boot_mean_diff']:.2f}%, 95% CI [{bt_b['ci95'][0]:+.2f}%, {bt_b['ci95'][1]:+.2f}%], p={bt_b['p_value_two_sided']:.2f} (NS)\n"
        f"• testC: ΔAP = +{bt_c['boot_mean_diff']:.2f}%, 95% CI [{bt_c['ci95'][0]:+.2f}%, {bt_c['ci95'][1]:+.2f}%], p={bt_c['p_value_two_sided']:.2f} (NS)"
    )
    ax1.text(0.02, 0.80, note_txt, transform=ax1.transAxes, fontsize=7.3,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef9e7', edgecolor='#b7950b', alpha=0.94))

    ax1.set_xticks(x)
    ax1.set_xticklabels([it[0].replace('\n', ' ') for it in items], fontsize=7.2, fontweight='bold', rotation=24, ha='right')
    ax1.set_ylim(64.5, 93.5)
    ax1.set_ylabel('Strict FP32 COCO AP50–95 (%)', fontsize=9.8, fontweight='bold')
    ax1.set_title('(a) SciDB Exploratory Benchmarks: All Baselines, Negative Probes & Paired Bootstrap', fontsize=10.2, fontweight='bold', pad=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.35)
    ax1.legend(loc='upper left', fontsize=7.4, ncol=3)

    # Subplot (b): Same-Geometry Background Intervention (Pure_FrozenDINO_CLS vs Pure_BGOPTA_S3, alpha=0)
    conds = [
        ('1.25x\nGrayOuter', 'C2_125_GrayOuter'),
        ('1.25x\nRealBG', 'C1_125_RealBG'),
        ('1.50x\nGrayOuter', 'C4_150_GrayOuter'),
        ('1.50x\nRealBG', 'C3_150_RealBG'),
        ('1.50x\nWhiteGlare', 'C5_150_WhiteGlare245'),
        ('1.50x\nShuffledBG', 'C6_150_ShuffledOuter'),
    ]
    xc = np.arange(len(conds))
    wc = 0.20
    s_series = [
        ('Pure DINO [CLS] (testB)', '#2980b9', [a1[k]['Pure_FrozenDINO_CLS']['testB']['AP50_95_mean'] for _, k in conds]),
        ('Pure BG-OPTA S3 (testB)', '#85c1e9', [a1[k]['Pure_BGOPTA_S3_NoYOLOFuse']['testB']['AP50_95_mean'] for _, k in conds]),
        ('Pure DINO [CLS] (testC)', '#117a65', [a1[k]['Pure_FrozenDINO_CLS']['testC']['AP50_95_mean'] for _, k in conds]),
        ('Pure BG-OPTA S3 (testC)', '#f39c12', [a1[k]['Pure_BGOPTA_S3_NoYOLOFuse']['testC']['AP50_95_mean'] for _, k in conds]),
    ]
    for si, (slbl, scol, svals) in enumerate(s_series):
        pos = xc + (si - 1.5) * wc
        rs = ax2.bar(pos, svals, width=wc, color=scol, edgecolor='black', linewidth=0.55, label=slbl)
        for r in rs:
            hv = r.get_height()
            ax2.text(r.get_x() + r.get_width() / 2.0, hv + 0.4, f"{hv:.1f}", ha='center', va='bottom', fontsize=6.2, fontweight='bold', rotation=60)

    decomp_txt = (
        "Pure Frozen DINO [CLS] Decomposition (1.25x -> 1.50x):\n"
        "• Resolution Loss (1.25x Gray -> 1.50x Gray): -2.08pt (testB), -2.02pt (testC)\n"
        "• Same-Geom Real BG (1.50x Gray -> 1.50x Real): -0.68pt (testB), -2.56pt (testC)\n"
        "• BG-OPTA helps on Shuffled Clutter (+6.42pt), hurts on Uniform Gray/White (-2.94pt)"
    )
    ax2.text(0.02, 0.03, decomp_txt, transform=ax2.transAxes, fontsize=7.2,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#566573', alpha=0.93))

    ax2.set_xticks(xc)
    ax2.set_xticklabels([c[0] for c in conds], fontsize=8.3, fontweight='bold')
    ax2.set_ylim(54.0, 90.5)
    ax2.set_ylabel('Pure Stage-2 Classifier AP50–95 (%, alpha_det = 0)', fontsize=9.6, fontweight='bold')
    ax2.set_title('(b) Same-Geometry Background Intervention: Resolution Drop vs BG Interference', fontsize=10.2, fontweight='bold', pad=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.35)
    ax2.legend(loc='upper right', fontsize=7.6, ncol=2)

    fig.tight_layout(pad=1.5)
    f1_p = FIG_DIR / 'fig1_complete_baselines_negatives_and_bootstrap.png'
    fig.savefig(f1_p, dpi=300)
    plt.close(fig)

    # ==========================================================================
    # Figure 2: (a) Fair Post-Fusion + Post-NMS End-to-End Latency (val & testC)
    #           (b) External & Cross-Source Benchmarks Including All Negative Results
    # ==========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 5.8), dpi=300)

    lat_modes = [
        ('E1: Native YOLOv8n', 'E1_Native_YOLOv8n',
         0.5 * (tcd_agg['B0_YOLOv8n_Native']['testB']['AP50_95_mean'] + tcd_agg['B0_YOLOv8n_Native']['testC']['AP50_95_mean']), '#7f8c8d', 'o', (8, -6)),
        ('E2: Full DINOv2 (1-Crop)', 'E2_Full_DINOv2_1Crop_Hard',
         0.5 * (tcd_agg['B1a_Full_DINOv2_GrayPad_Orig']['testB']['AP50_95_mean'] + tcd_agg['B1a_Full_DINOv2_GrayPad_Orig']['testC']['AP50_95_mean']), '#e74c3c', 's', (-118, -22)),
        ('E3: Abl1 (1-Crop+14Cls+NMS)', 'E3_Abl1_SingleCrop_14Cls_Calib',
         0.5 * (np.mean(pt_a2['testB']['Abl1_SingleCrop_14Cls_Calib']) + np.mean(pt_a2['testC']['Abl1_SingleCrop_14Cls_Calib'])), '#2980b9', 'D', (10, -18)),
        ('E4: BG-OPTA 1.5x (Neg)', 'E4_BGOPTA_1Pass_Full',
         0.5 * (scidb_p2['Ours_S3_BGOPTA_1Pass']['testB']['mean'] + scidb_p2['Ours_S3_BGOPTA_1Pass']['testC']['mean']), '#e67e22', 'X', (8, -12)),
        ('E7: Abl3 (3-Crop Static+NMS)', 'E7_Abl3_TriGranular_StaticWeight',
         0.5 * (np.mean(pt_a2['testB']['Abl3_Diff_StaticWeight']) + np.mean(pt_a2['testC']['Abl3_Diff_StaticWeight'])), '#117a65', '*', (-128, 10)),
        ('E5: Abl2 (Disagree Esc+NMS)', 'E5_Abl2_Disagreement_Esc',
         0.5 * (tcd_agg['Abl2_Disagreement_Esc_Only']['testB']['AP50_95_mean'] + tcd_agg['Abl2_Disagreement_Esc_Only']['testC']['AP50_95_mean']), '#5dade2', '^', (-115, -34)),
        ('E6: TCD-MGE v2 (TriSig+NMS)', 'E6_TCDMGE_v2_TriSignal_Esc',
         0.5 * (tcd_agg['Ours_TCDMGE_Selective']['testB']['AP50_95_mean'] + tcd_agg['Ours_TCDMGE_Selective']['testC']['AP50_95_mean']), '#8e44ad', 'v', (-86, 13)),
    ]

    for lbl_e, ek, ap_m, col, mk, offs in lat_modes:
        ms_tc = a4[ek]['testC_all602imgs']['mean_ms']
        ms_va = a4[ek]['val_200imgs']['mean_ms']
        p95_va = a4[ek]['val_200imgs']['p95_ms']
        sz = 180 if mk == '*' else 95
        ax1.scatter(ms_tc, ap_m, s=sz, color=col, marker=mk, edgecolors='black', linewidths=0.85, zorder=6)
        ax1.annotate(f"{lbl_e}\ntestC={ms_tc:.1f}ms | val={ms_va:.1f}ms (P95 {p95_va:.0f}ms)",
                     xy=(ms_tc, ap_m), xytext=offs, textcoords='offset points', fontsize=7.2,
                     arrowprops=dict(arrowstyle='-', color=col, lw=0.7, alpha=0.6),
                     bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor=col, alpha=0.92))

    ax1.axvline(a4['E2_Full_DINOv2_1Crop_Hard']['testC_all602imgs']['mean_ms'], color='#e74c3c', linestyle=':', linewidth=1.2,
                label='Full DINOv2 testC Mean (14.64 ms) — All Multi-Crop Methods Are Slower')
    ax1.set_xlabel('Complete End-to-End Latency Including Fusion + Re-Weighting + NMS on testC (ms)', fontsize=9.4, fontweight='bold')
    ax1.set_ylabel('Mean Exploratory AP50–95 on testB & testC (%)', fontsize=9.6, fontweight='bold')
    ax1.set_title('(a) Fair End-to-End Speed–Accuracy Audit (RTX 4060 Ti, Strict FP32, N=602)', fontsize=10.2, fontweight='bold', pad=10)
    ax1.set_xlim(6.5, 29.5)
    ax1.set_ylim(70.5, 84.8)
    ax1.grid(True, linestyle='--', alpha=0.35)
    ax1.legend(loc='lower right', fontsize=7.8)

    # Subplot (b): External & Cross-Source Comparison Showing ALL Strong-Baseline Wins & Negatives
    zen_bg = p2['模块C_修复版Zenodo外部果园零丢框探索评测']['summary']
    zen_mg = p2b['冻结多粒度模型外部与最终盲测汇总']['zenodo_repaired_zerodrop']
    uat_bg = p2['模块D_CitrusUAT真实叶片掩膜前景聚焦与跨域诊断']['cross_domain_5cls_summary']
    uat_mg = p2b['冻结多粒度模型外部与最终盲测汇总']['citrusuat_5cls']
    vlt_bg = p2['模块E_未触碰SweetOrange最终封口盲测']['summary']
    vlt_mg = p2b['冻结多粒度模型外部与最终盲测汇总']['vault_sweet_orange_1389']

    ext_groups = [
        'Zenodo (Exploratory)\nZero-Drop 2Cls AP50',
        'Zenodo (Exploratory)\nZero-Drop 3Cls AP50',
        'CitrusUAT (Studio)*\n5-Cls Macro-F1',
        'SweetOrange (v1-Used)\nOpen 14-Cls Hit',
        'SweetOrange (v1-Used)\n3-Subset Macro-F1',
    ]
    xe = np.arange(len(ext_groups))
    we = 0.15
    ext_models = [
        ('B0: Native YOLOv8n', '#7f8c8d', [
            zen_bg['B0_YOLOv8n_Native']['det_2cls_AP50']['mean'],
            zen_bg['B0_YOLOv8n_Native']['det_3cls_repaired_AP50']['mean'],
            float(uat_bg['B0_YOLOv8n']['aligned_5cls_f1']['mean_std'].split('±')[0]),
            float(vlt_bg['B0_YOLOv8n_Native']['strict_14cls_hit']['mean_std'].split('±')[0]),
            float(vlt_bg['B0_YOLOv8n_Native']['aligned_3cls_f1']['mean_std'].split('±')[0]),
        ]),
        ('B1: Full DINOv2 (1-Crop)', '#e74c3c', [
            zen_bg['B1_Full_DINOv2_1Crop']['det_2cls_AP50']['mean'],
            zen_bg['B1_Full_DINOv2_1Crop']['det_3cls_repaired_AP50']['mean'],
            float(uat_bg['B1_Full_DINOv2']['aligned_5cls_f1']['mean_std'].split('±')[0]),
            float(vlt_bg['B1_Full_DINOv2_1Crop']['strict_14cls_hit']['mean_std'].split('±')[0]),
            float(vlt_bg['B1_Full_DINOv2_1Crop']['aligned_3cls_f1']['mean_std'].split('±')[0]),
        ]),
        ('BG-OPTA 1.5x (Probe)', '#f39c12', [
            zen_bg['Ours_S3_BGOPTA_1Pass']['det_2cls_AP50']['mean'],
            zen_bg['Ours_S3_BGOPTA_1Pass']['det_3cls_repaired_AP50']['mean'],
            float(uat_bg['Ours_S3_BGOPTA']['aligned_5cls_f1']['mean_std'].split('±')[0]),
            float(vlt_bg['Ours_S3_BGOPTA_1Pass']['strict_14cls_hit']['mean_std'].split('±')[0]),
            float(vlt_bg['Ours_S3_BGOPTA_1Pass']['aligned_3cls_f1']['mean_std'].split('±')[0]),
        ]),
        ('Abl2: Disagree Esc', '#3498db', [
            zen_mg['Abl2_Disagreement_Esc']['det_2cls_AP50']['mean'],
            zen_mg['Abl2_Disagreement_Esc']['det_3cls_repaired_AP50']['mean'],
            uat_mg['Abl2_Disagreement_Esc']['aligned_5cls_f1']['mean'],
            vlt_mg['Abl2_Disagreement_Esc']['strict_14cls_hit']['mean'],
            vlt_mg['Abl2_Disagreement_Esc']['aligned_3cls_f1']['mean'],
        ]),
        ('Abl3: 3-Crop Diff Static', '#117a65', [
            zen_mg['Abl3_Full_3Crop_Static']['det_2cls_AP50']['mean'],
            zen_mg['Abl3_Full_3Crop_Static']['det_3cls_repaired_AP50']['mean'],
            uat_mg['Abl3_Full_3Crop_Static']['aligned_5cls_f1']['mean'],
            vlt_mg['Abl3_Full_3Crop_Static']['strict_14cls_hit']['mean'],
            vlt_mg['Abl3_Full_3Crop_Static']['aligned_3cls_f1']['mean'],
        ]),
    ]

    for mi, (mlbl, mcol, mvals) in enumerate(ext_models):
        pos = xe + (mi - 2.0) * we
        rs = ax2.bar(pos, mvals, width=we, color=mcol, edgecolor='black', linewidth=0.55, label=mlbl)
        for r in rs:
            hv = r.get_height()
            ax2.text(r.get_x() + r.get_width() / 2.0, hv + 0.6, f"{hv:.1f}", ha='center', va='bottom', fontsize=6.1, fontweight='bold', rotation=55)

    caveat_txt = (
        "*Key Negative Findings & Audit Caveats:\n"
        "1) On CitrusUAT studio gray board, Full DINOv2 (73.4% F1) beats Abl3 (69.9%) & BG-OPTA (64.4%) due to YOLOv8n drag (45.9%);\n"
        "   CitrusUAT leaf mask in Phase 2 was a gray-border color-threshold heuristic pseudo-mask, NOT human GT segmentation.\n"
        "2) On SweetOrange V1 (previously touched in v1 cls), B0 YOLOv8n wins 3-subset F1 (80.2%) despite lowest Open 14-Cls hit (54.9%)."
    )
    ax2.text(0.015, 0.80, caveat_txt, transform=ax2.transAxes, fontsize=6.6,
             bbox=dict(boxstyle='round,pad=0.26', facecolor='#fdedec', edgecolor='#c0392b', alpha=0.94))

    ax2.set_xticks(xe)
    ax2.set_xticklabels(ext_groups, fontsize=8.0, fontweight='bold')
    ax2.set_ylim(0, 112.0)
    ax2.set_ylabel('Score (%, 3-Seed Mean)', fontsize=9.6, fontweight='bold')
    ax2.set_title('(b) Exploratory External & Single-Leaf Benchmarks (Including Baseline Wins & Negatives)', fontsize=10.0, fontweight='bold', pad=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.35)
    ax2.legend(loc='upper right', fontsize=7.1, ncol=2)

    fig.tight_layout(pad=1.5)
    f2_p = FIG_DIR / 'fig2_fair_e2e_speed_and_external_with_negatives.png'
    fig.savefig(f2_p, dpi=300)
    plt.close(fig)
    print(f"[诚实审计配图生成完毕]\n  1) {f1_p}\n  2) {f2_p}")


if __name__ == '__main__':
    main()
