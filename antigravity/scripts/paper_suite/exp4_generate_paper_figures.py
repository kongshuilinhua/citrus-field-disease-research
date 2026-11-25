# -*- coding: utf-8 -*-
"""
实验 E4: 生成 SCI 小论文高分辨率插图 (300 DPI PNG, 零遮挡出版级排版)
1. fig1_pareto_ap_latency.png: 跨场景精度 (testC AP50-95) vs 真实单图推理延迟 (ms) 帕累托前沿 + 零监督域偏移自适应触发率对比图
2. fig2_symtex_ablation_and_external.png: SymTex 组件级 3 种子消融 (零额外参数 3.347M) + 多源外部公开集零样本泛化与叶片前景能量占比图
3. fig3_external_orchard_blind_det.png: 外部真实果园带框检测集 (Zenodo 11148332, 519图/1765框) 零样本检测 AP50 与 3 类病理拓扑框级分类对比图
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(r'D:\新版柑橘')
EXP_DIR = ROOT / '多路线创新筛选_20261124' / '小论文完整实验_20261124'
RES_DIR = EXP_DIR / 'results'
FIG_DIR = EXP_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    e1 = json.loads((RES_DIR / 'e1_symtex_ablation_and_kd_summary.json').read_text(encoding='utf-8'))['结果汇总']
    e2_full = json.loads((RES_DIR / 'e2_end2end_mgs_symtex_summary.json').read_text(encoding='utf-8'))
    e2 = e2_full['三种子汇总']
    lat = e2_full['实测单图延迟_ms']
    e3 = json.loads((RES_DIR / 'e3_external_blind_validation_summary.json').read_text(encoding='utf-8'))

    # ---------------- Figure 1: Latency vs Cross-Scene AP Pareto + Self-Adaptive Trigger Rate ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.4), dpi=300)

    pts = [
        ('YOLOv8n Native (Single-Stage)', lat['native']['shifted_testC']['mean_ms'], e2['S0_YOLOv8n_Native']['testC']['AP50_95_mean'], '#7f8c8d', 'o', (8, 5)),
        ('Full Cascade + Vanilla MBV3', lat['full_light']['shifted_testC']['mean_ms'], e2['S1_Full_Vanilla_MBV3']['testC']['AP50_95_mean'], '#e74c3c', 's', (-88, 8)),
        ('Full Cascade + SymTex (No KD)', lat['full_light']['shifted_testC']['mean_ms'], e2['S2a_Full_SymTex_NoKD']['testC']['AP50_95_mean'], '#f39c12', 'D', (-92, 12)),
        ('MGS + SymTex-Pruned (3.35M, No KD)', lat['mgs_light']['shifted_testC']['mean_ms'], e2['Ours_Light_MGS_SymTex_NoKD']['testC']['AP50_95_mean'], '#27ae60', '^', (8, -6)),
        ('MGS + Distilled Light (3.35M)', lat['mgs_light']['shifted_testC']['mean_ms'], e2['Ours_Light_MGS_VanillaKD']['testC']['AP50_95_mean'], '#16a085', 'P', (-98, -16)),
        ('Ours-Pro (MGS + DINOv2)', lat['mgs_pro']['shifted_testC']['mean_ms'], e2['Ours_Pro_MGS_DINOv2']['testC']['AP50_95_mean'], '#2980b9', '*', (8, -8)),
    ]
    for label, x_ms, y_ap, col, mk, offset in pts:
        sz = 200 if '*' in mk else 115
        ax1.scatter(x_ms, y_ap, s=sz, c=col, marker=mk, label=label, zorder=4, edgecolors='black', linewidths=0.7)
        ax1.annotate(f"{y_ap:.2f}% ({x_ms:.1f}ms)", (x_ms, y_ap), textcoords="offset points", xytext=offset, fontsize=8.3, fontweight='bold')

    ax1.set_xlim(7.7, 17.6)
    ax1.set_ylim(71.5, 83.2)
    ax1.set_xlabel('Single-Image End-to-End Latency on RTX 4060 Ti (ms)', fontsize=10.5)
    ax1.set_ylabel('Cross-Background Detection AP50-95 on TestC (%)', fontsize=10.5)
    ax1.set_title('(a) Accuracy-Latency Pareto Frontier (3-Seed Mean)', fontsize=11.5, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left', fontsize=8.2, framealpha=0.92)

    splits_lbl = ['Source Val\n(In-Domain)', 'TestB\n(Orchard Shift)', 'TestC\n(Board/Exposure Shift)']
    x = np.arange(len(splits_lbl))
    w = 0.25
    ap_nat = [e2['S0_YOLOv8n_Native'][s]['AP50_95_mean'] for s in ['val', 'testB', 'testC']]
    ap_light = [e2['Ours_Light_MGS_SymTex_NoKD'][s]['AP50_95_mean'] for s in ['val', 'testB', 'testC']]
    ap_pro = [e2['Ours_Pro_MGS_DINOv2'][s]['AP50_95_mean'] for s in ['val', 'testB', 'testC']]
    trig_rates = [e2['Ours_Pro_MGS_DINOv2'][s]['trig_pct_mean'] for s in ['val', 'testB', 'testC']]

    ax2.bar(x - w, ap_nat, width=w, label='YOLOv8n Native', color='#95a5a6', edgecolor='black', linewidth=0.6)
    ax2.bar(x, ap_light, width=w, label='Ours-Light (MGS + SymTex 3.35M)', color='#2ecc71', edgecolor='black', linewidth=0.6)
    ax2.bar(x + w, ap_pro, width=w, label='Ours-Pro (MGS + DINOv2)', color='#3498db', edgecolor='black', linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits_lbl, fontsize=9.5)
    ax2.set_ylim(60, 97)
    ax2.set_ylabel('COCO AP50-95 (%)', fontsize=10.5)
    ax2.set_title('(b) End-to-End Detection & Self-Adaptive Triggering', fontsize=11.5, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

    ax2_r = ax2.twinx()
    ax2_r.plot(x, trig_rates, color='#d35400', marker='o', linewidth=2.2, linestyle='--', label='MGS Trigger Rate (%)')
    for xi, tr in zip(x, trig_rates):
        ax2_r.annotate(f"Trig: {tr:.1f}%", (xi, tr), textcoords="offset points", xytext=(0, 8), ha='center', color='#d35400', fontsize=8.8, fontweight='bold')
    ax2_r.set_ylim(0, 72)
    ax2_r.set_ylabel('Auto-Adapted Trigger Rate (%)', color='#d35400', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8.2, framealpha=0.92)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_pareto_ap_latency.png', dpi=300)
    plt.close(fig)

    # ---------------- Figure 2: SymTex Component Ablation & Multi-Dataset Blind Generalization ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.3), dpi=300)
    abl_keys = ['B1_Vanilla_Pruned', 'Abl_BASD_Only', 'Abl_IIHFR_Only', 'SymTex_Pruned', 'Vanilla_KD', 'Ours_SymTex_KD_Pruned']
    abl_names = ['Vanilla\n(3.35M)', '+BASD\nOnly', '+II-HFR\nOnly', 'SymTex\n(Pruned)', 'Vanilla\nKD', 'SymTex-KD\n(3.35M)']
    tb_f1 = [e1[k]['testB_f1'][1] for k in abl_keys]
    tc_f1 = [e1[k]['testC_f1'][1] for k in abl_keys]
    eim = [e1[k]['uat_eim_pct'][1] for k in abl_keys]

    xa = np.arange(len(abl_keys))
    bw = 0.35
    ax1.bar(xa - bw / 2, tb_f1, width=bw, label='SciDB TestB Macro-F1', color='#5dade2', edgecolor='black', linewidth=0.6)
    ax1.bar(xa + bw / 2, tc_f1, width=bw, label='SciDB TestC Macro-F1', color='#48c9b0', edgecolor='black', linewidth=0.6)
    ax1.set_xticks(xa)
    ax1.set_xticklabels(abl_names, fontsize=8.5)
    ax1.set_ylim(78.5, 86.8)
    ax1.set_ylabel('Macro-F1 (%)', fontsize=10.5)
    ax1.set_title('(a) 3-Seed Component Ablation (Zero Extra Params)', fontsize=11.5, fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left', fontsize=8.5)

    ext_cls = e3['外部独立分类盲测']
    sw_vals = [ext_cls[k]['SweetOrange_F1_mean'] for k in abl_keys]
    zn_vals = [ext_cls[k]['Zenodo_F1_mean'] for k in abl_keys]
    pm_vals = [ext_cls[k]['Pomelo_F1_mean'] for k in abl_keys]
    ax2.plot(xa, sw_vals, marker='s', linewidth=2.0, color='#8e44ad', label='SweetOrange V1 (N=1135) F1')
    ax2.plot(xa, zn_vals, marker='^', linewidth=2.0, color='#e67e22', label='Zenodo Citrus (N=560) F1')
    ax2.plot(xa, pm_vals, marker='D', linewidth=1.8, color='#2980b9', label='Pomelo V2 Cross-Species (N=292) F1')
    ax2.plot(xa, eim, marker='o', linewidth=1.8, linestyle=':', color='#27ae60', label='CitrusUAT Leaf-Mask Energy (%)')
    ax2.set_xticks(xa)
    ax2.set_xticklabels(abl_names, fontsize=8.5)
    ax2.set_ylim(58, 88)
    ax2.set_ylabel('Score / Percentage (%)', fontsize=10.5)
    ax2.set_title('(b) Zero-Shot External Blind Generalization & Leaf Focus', fontsize=11.5, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='lower right', fontsize=8.0, framealpha=0.92)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_symtex_ablation_and_external.png', dpi=300)
    plt.close(fig)

    # ---------------- Figure 3: Unseen Orchard Detection Blind Test (Zenodo 519 imgs / 1765 boxes) ----------------
    det_ext = e3['外部真实果园检测盲测']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2), dpi=300)

    d_keys = ['YOLOv8n_Native', 'Full_Vanilla_MBV3', 'MGS_SymTex_NoKD', 'Ours_Light_MGS_SymTex_KD', 'LowConf_DINOv2', 'Ours_Pro_MGS_DINOv2']
    d_lbls = ['YOLOv8n\nNative', 'Full\nVanilla', 'MGS+SymTex\n(No KD)', 'MGS+SymTexKD\n(Ours-Light)', 'LowConf\n+DINOv2', 'Ours-Pro\n(MGS+DINOv2)']
    ap2 = [det_ext['end2end_det_2cls_summary'][k]['AP50_mean'] for k in d_keys]
    ap3 = [det_ext['end2end_det_3cls_summary'][k]['AP50_mean'] for k in d_keys]

    xd = np.arange(len(d_keys))
    ax1.bar(xd - 0.18, ap2, width=0.35, label='2-Class Orchard Detection AP50', color='#5dade2', edgecolor='black', linewidth=0.6)
    ax1.bar(xd + 0.18, ap3, width=0.35, label='3-Class Fine Topology Detection AP50', color='#2ecc71', edgecolor='black', linewidth=0.6)
    ax1.set_xticks(xd)
    ax1.set_xticklabels(d_lbls, fontsize=8.3)
    ax1.set_ylim(18, 46.5)
    ax1.set_ylabel('Zero-Shot COCO AP50 (%)', fontsize=10.5)
    ax1.set_title('(a) External Orchard Blind Detection (519 Imgs, 1765 Boxes)', fontsize=11.0, fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left', fontsize=8.4, framealpha=0.92)

    gt3_f1 = [det_ext['gt_box_3cls_topology_classification'][k]['F1_mean'] for k in abl_keys]
    bars = ax2.bar(xa, gt3_f1, width=0.52, color=['#95a5a6', '#85c1e9', '#76d7c4', '#48c9b0', '#f5b041', '#27ae60'], edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, gt3_f1):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.2f}%", ha='center', va='bottom', fontsize=8.6, fontweight='bold')
    ax2.set_xticks(xa)
    ax2.set_xticklabels(abl_names, fontsize=8.5)
    ax2.set_ylim(50, 64.5)
    ax2.set_ylabel('3-Class Macro-F1 (%)', fontsize=10.5)
    ax2.set_title('(b) Orchard 1765-Box Topology Classification (Healthy/Mottling/Deficiency)', fontsize=10.5, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_external_orchard_blind_det.png', dpi=300)
    plt.close(fig)
    print(f"[完成] 零遮挡出版级论文配图已更新于: {FIG_DIR}", flush=True)


if __name__ == '__main__':
    main()
