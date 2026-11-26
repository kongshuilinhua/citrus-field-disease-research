# -*- coding: utf-8 -*-
"""
生成两层八候选预注册方法库与保形风险控制全景对比图 (generate_candidate_arsenal_figures.py)
严格包含全部基线与负向对照（含室内单叶 CitrusUAT 强基线表现与 PCR 最小置信度负结果对照），零文字与误差棒重叠。
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(r'D:\新版柑橘\citrus-field-disease-research\antigravity\v4_orchard_candidate_arsenal')
RES_JSON = ROOT / 'results' / 'orchard_candidate_arsenal_complete.json'
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_fig1_layer1_candidates(data: dict):
    scidb = data['layer1_zero_shot_detection_candidates_scidb']
    cross = data['layer1_cross_domain_zenodo_and_citrusuat']
    zen = cross['Zenodo_11148332_519imgs']
    uat = cross['CitrusUAT_500imgs_IndoorSingleLeaf_OOD']

    keys_l1 = [
        ('B0_Native_YOLOv8n', 'B0: YOLOv8n\n(0 ViT)'),
        ('B1_Full_DINOv2_1Crop', 'B1: Full DINOv2\n(1.0x 硬覆盖)'),
        ('B2_Static_Prob_Fuse', 'B2: 静态概率融合\n(1.0x 旧式)'),
        ('C1_Fast1Crop_StaticCalib', 'C1: 单框对数校准\n(1.0x 边缘默认)'),
        ('C2_Fast1Crop_OOD_CertaintyPool', 'C2: OOD流形确信池\n(1.0x 防退化)'),
        ('C3_ZeroMLP_DualCrop_CoreStd', 'C3: 零MLP双尺度\n(2.0x 跨品种)'),
        ('C4_HardBudget25_MultiScale', 'C4: 25%硬预算升级\n(1.5x 控时)'),
        ('C5a_Server_3Crop_PlainConcat', 'C5: 三尺度集成\n(3.0x 云端预标)'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.2, 6.8), gridspec_kw={'width_ratios': [1.32, 1.0]})
    fig.subplots_adjust(wspace=0.23, bottom=0.22, top=0.85)

    x = np.arange(len(keys_l1))
    w = 0.37
    b_means = [scidb[k]['testB']['AP50_95_mean'] for k, _ in keys_l1]
    b_stds = [scidb[k]['testB']['AP50_95_std'] for k, _ in keys_l1]
    c_means = [scidb[k]['testC']['AP50_95_mean'] for k, _ in keys_l1]
    c_stds = [scidb[k]['testC']['AP50_95_std'] for k, _ in keys_l1]

    bars_b = ax1.bar(x - w / 2, b_means, w, yerr=b_stds, capsize=3.2, color='#3B82F6', alpha=0.88,
                     error_kw={'lw': 1.2, 'ecolor': '#1E3A8A'}, label='SciDB testB (自然田间复杂背景)')
    bars_c = ax1.bar(x + w / 2, c_means, w, yerr=c_stds, capsize=3.2, color='#10B981', alpha=0.88,
                     error_kw={'lw': 1.2, 'ecolor': '#064E3B'}, label='SciDB testC (高光白板异质背景)')

    ax1.axhline(scidb['B1_Full_DINOv2_1Crop']['testB']['AP50_95_mean'], color='#1D4ED8', ls='--', lw=1.1, alpha=0.6)
    ax1.axhline(scidb['B1_Full_DINOv2_1Crop']['testC']['AP50_95_mean'], color='#047857', ls='--', lw=1.1, alpha=0.6)

    for rect, val, sd in zip(bars_b, b_means, b_stds):
        ax1.text(rect.get_x() + rect.get_width() / 2, val + sd + 0.45, f'{val:.1f}',
                 ha='center', va='bottom', fontsize=8.3, color='#1E3A8A', fontweight='bold')
    for rect, val, sd in zip(bars_c, c_means, c_stds):
        ax1.text(rect.get_x() + rect.get_width() / 2, val + sd + 0.45, f'{val:.1f}',
                 ha='center', va='bottom', fontsize=8.3, color='#065F46', fontweight='bold')

    ax1.set_ylim(63, 92.5)
    ax1.set_ylabel('COCO AP50-95 (%, 3随机种子均值±标准差)', fontsize=10.5)
    ax1.set_title('(a) 第一层：5个零样本预注册检测候选 (C1-C5) vs 三类基线 (B0-B2)', fontsize=11.3, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in keys_l1], fontsize=8.8, rotation=22, ha='right')
    ax1.grid(axis='y', ls=':', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=9.2, frameon=True)

    keys_cd = [
        ('B0_Native_YOLOv8n', 'B0: YOLOv8n'),
        ('B1_Full_DINOv2_1Crop', 'B1: Full DINOv2'),
        ('C1_Fast1Crop_StaticCalib', 'C1: 静态对数校准'),
        ('C2_Fast1Crop_OOD_CertaintyPool', 'C2: OOD流形确信池'),
        ('C3_ZeroMLP_DualCrop_CoreStd', 'C3: 零MLP双尺度'),
        ('C5a_Server_3Crop_PlainConcat', 'C5: 三尺度集成'),
    ]
    x2 = np.arange(len(keys_cd))
    z_m = [zen[k]['3cls_AP50']['mean'] for k, _ in keys_cd]
    z_s = [zen[k]['3cls_AP50']['std'] for k, _ in keys_cd]
    u_m = [uat[k]['MacroF1_5cls']['mean'] for k, _ in keys_cd]
    u_s = [uat[k]['MacroF1_5cls']['std'] for k, _ in keys_cd]

    b_z = ax2.bar(x2 - w / 2, z_m, w, yerr=z_s, capsize=3.2, color='#F59E0B', alpha=0.90,
                  error_kw={'lw': 1.2, 'ecolor': '#78350F'}, label='Zenodo 外部果园 (3类零丢框 AP50 %)')
    b_u = ax2.bar(x2 + w / 2, u_m, w, yerr=u_s, capsize=3.2, color='#8B5CF6', alpha=0.88,
                  error_kw={'lw': 1.2, 'ecolor': '#4C1D95'}, label='CitrusUAT 室内单叶 OOD (5类 Macro-F1 %)')

    for rect, val, sd in zip(b_z, z_m, z_s):
        ax2.text(rect.get_x() + rect.get_width() / 2, val + sd + 0.7, f'{val:.1f}',
                 ha='center', va='bottom', fontsize=8.4, color='#92400E', fontweight='bold')
    for rect, val, sd in zip(b_u, u_m, u_s):
        ax2.text(rect.get_x() + rect.get_width() / 2, val + sd + 0.7, f'{val:.1f}',
                 ha='center', va='bottom', fontsize=8.4, color='#5B21B6', fontweight='bold')

    ax2.set_ylim(20, 92.0)
    ax2.set_ylabel('跨数据集评测指标 (%, 3种子均值±标准差)', fontsize=10.5)
    ax2.set_title('(b) 跨域果园 (Zenodo) 与室内近景单叶 OOD 防护验证 (含强基线反超对照)', fontsize=11.3, fontweight='bold', pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for _, lbl in keys_cd], fontsize=8.9, rotation=22, ha='right')
    ax2.grid(axis='y', ls=':', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=9.0, frameon=True)

    fig.suptitle('Antigravity v4 零作假预注册检测候选方法库：精度梯度与跨域 OOD 防护特性', fontsize=13.5, fontweight='bold', y=0.96)
    out_p = FIG_DIR / 'fig1_layer1_candidate_arsenal_and_ood_guard.png'
    fig.savefig(out_p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_p}")


def plot_fig2_layer2_conformal_and_triage(data: dict):
    crc = data['layer2_R2_conformal_risk_control']['testB']
    pcr_g1 = data['layer2_R1_visual_and_R3_pcr_triage']['pcr_597_suspect_triage']['G1_Strict4Symptoms']['disjoint_calibration_triage_curves']
    vis_b = data['layer2_R1_visual_and_R3_pcr_triage']['visual_symptom_triage']['testB']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.8, 6.7))
    fig.subplots_adjust(wspace=0.23, bottom=0.16, top=0.83)

    crc_models = [
        ('B0_Native_YOLOv8n', 'B0: YOLOv8n', '#94A3B8'),
        ('B1_Full_DINOv2_1Crop', 'B1: Full DINOv2', '#64748B'),
        ('C1_Fast1Crop_StaticCalib', 'C1: 单框对数校准', '#3B82F6'),
        ('C2_Fast1Crop_OOD_CertaintyPool', 'C2: OOD 流形确信池', '#0EA5E9'),
        ('C3_ZeroMLP_DualCrop_CoreStd', 'C3: 零MLP双尺度', '#10B981'),
        ('C5a_Server_3Crop_PlainConcat', 'C5: 三尺度集成', '#F59E0B'),
    ]
    alphas = ['alpha_0.10', 'alpha_0.15', 'alpha_0.20']
    alpha_lbls = ['目标风险上限 α=10%\n(严苛自动放行档)', '目标风险上限 α=15%\n(常规自动放行档)', '目标风险上限 α=20%\n(宽松自动放行档)']
    x = np.arange(len(alphas))
    bw = 0.132

    for idx, (mk, mlbl, col) in enumerate(crc_models):
        covs = [crc[mk]['n_cal_80'][ak]['auto_coverage_pct_3seed_mean'] for ak in alphas]
        r10 = crc[mk]['n_cal_80']['alpha_0.10']['empirical_risk_3seed_mean'] * 100.0
        pos = x + (idx - 2.5) * bw
        rects = ax1.bar(pos, covs, bw, color=col, alpha=0.91, label=f'{mlbl} (α=10%实测风险 {r10:.1f}%)')
        for r, cv in zip(rects, covs):
            ax1.text(r.get_x() + r.get_width() / 2, cv + 0.65, f'{cv:.1f}',
                     ha='center', va='bottom', fontsize=7.3, fontweight='bold', rotation=35)

    ax1.set_ylim(70, 114.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(alpha_lbls, fontsize=9.6)
    ax1.set_ylabel('保形风险控制下的自动放行覆盖率 Coverage (%)', fontsize=10.4)
    ax1.set_title('(a) R2 有限样本保形风险控制 (CRC, n_cal=80, 200次图像互斥切分)\nα=10% 下候选 C1-C5 将自动放行率从 80.0% 提升至 93.3%-94.7%', fontsize=10.6, fontweight='bold', pad=11)
    ax1.grid(axis='y', ls=':', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=8.2, ncol=2, frameon=True, framealpha=0.94)

    budgets_num = [5, 10, 20, 30]
    b_keys = ['5%', '10%', '20%', '30%']

    y_rand = budgets_num
    y_pcr_neg = [pcr_g1['ZeroCalib_MinConfidence_NegativeCtrl'][bk]['capture_pct_mean'] for bk in b_keys]
    y_pcr_b0 = [pcr_g1['HardClassPrior_B0_YOLOv8n_n100'][bk]['capture_pct_3seed_mean'] for bk in b_keys]
    y_pcr_c2 = [pcr_g1['R3_SoftPosteriorEB_C2_1Crop_n100'][bk]['capture_pct_3seed_mean'] for bk in b_keys]
    y_pcr_c5 = [pcr_g1['R3_SoftPosteriorEB_C5a_3Crop_n100'][bk]['capture_pct_3seed_mean'] for bk in b_keys]
    y_vis_p = [vis_b['Baseline_1MinusMaxProb'][bk]['capture_pct_mean'] for bk in b_keys]
    y_vis_r1 = [vis_b['R1_ZeroLabel_JS_Margin_Discordance'][bk]['capture_pct_mean'] for bk in b_keys]

    ax2.plot(budgets_num, y_rand, '--', color='#9CA3AF', lw=1.6, label='随机抽检基线 (1.0x)')
    ax2.plot(budgets_num, y_pcr_neg, 'v--', color='#EF4444', lw=1.8, ms=6.2, label='[负向对照] 最小置信度送审 PCR (5%预算仅6.6%)')
    ax2.plot(budgets_num, y_pcr_b0, 's-', color='#6366F1', lw=2.0, ms=6.2, label='DS 硬类别先验 (YOLOv8n, n_cal=100, PCR G1)')
    ax2.plot(budgets_num, y_pcr_c2, 'o-', color='#0EA5E9', lw=2.3, ms=6.8, label='R3 软后验经验贝叶斯 (C2单框, n_cal=100, PCR G1)')
    ax2.plot(budgets_num, y_pcr_c5, 'D-', color='#10B981', lw=2.1, ms=6.0, label='R3 软后验经验贝叶斯 (C5集成, n_cal=100, PCR G1)')
    ax2.plot(budgets_num, y_vis_r1, '^--', color='#D97706', lw=1.8, ms=6.2, label='R1b 跨模型JS分歧 (零标签视觉错判捕获, testB)')
    ax2.plot(budgets_num, y_vis_p, '*-', color='#F59E0B', lw=2.3, ms=8.0, label='R1a 校准后验不确定度 1-maxP (零标签视觉错判捕获, testB)')

    for bx, v_b0, v_c2, v_vp in zip(budgets_num, y_pcr_b0, y_pcr_c2, y_vis_p):
        ax2.annotate(f'{v_b0:.1f}%', (bx, v_b0), textcoords='offset points', xytext=(0, -13), ha='center', fontsize=7.9, color='#4338CA')
        ax2.annotate(f'{v_c2:.1f}%', (bx, v_c2), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8.0, color='#0369A1', fontweight='bold')
        ax2.annotate(f'{v_vp:.1f}%', (bx, v_vp), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8.0, color='#B45309', fontweight='bold')

    ax2.set_xlim(3, 32.5)
    ax2.set_ylim(0, 136)
    ax2.set_xticks(budgets_num)
    ax2.set_xticklabels(['5% 复核预算', '10% 复核预算', '20% 复核预算', '30% 复核预算'], fontsize=9.8)
    ax2.set_ylabel('目标错误样本捕获率 Error Capture Rate (%)', fontsize=10.4)
    ax2.set_title('(b) R1 零标签视觉纠错 vs R3 软后验经验贝叶斯 PCR 疑似症状送检\n(严格区分视觉错判与潜伏病原学错判；黄化仅作疑似筛查不等于确诊HLB)', fontsize=10.6, fontweight='bold', pad=11)
    ax2.grid(True, ls=':', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=8.1, frameon=True, framealpha=0.94)

    fig.suptitle('Antigravity v4 第二层决策候选库：有限样本保形风险控制 (R2) 与双场景分层复核策略 (R1 & R3)', fontsize=13.5, fontweight='bold', y=0.96)
    out_p = FIG_DIR / 'fig2_layer2_conformal_risk_and_pcr_triage.png'
    fig.savefig(out_p, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_p}")


def main():
    data = json.loads(RES_JSON.read_text(encoding='utf-8'))
    plot_fig1_layer1_candidates(data)
    plot_fig2_layer2_conformal_and_triage(data)


if __name__ == '__main__':
    main()
