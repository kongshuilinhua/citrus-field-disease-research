# Antigravity 独立研究归档（与 GPT、DeepSeek 成果严格物理隔离存放）

> **隔离归档说明**：本目录（`antigravity/`）专门存放 **Antigravity** 独立完成的方法诊断、代码实现与实验验证，与仓库内 **GPT / 协调者**（`coordination/`、`docs/项目现状_*.md`、`docs/research/`）及 **DeepSeek（DS）**（`docs/experiments/`）完全分目录独立归档，互不混写、互不覆盖。

---

## 一、主推 SCI 小论文定稿成果：`v3_bgopta_and_dtemop/`（回应全部审稿复核、零泄漏机理闭环终稿版）

针对前期 `v2_tcd_mge` 复核中指出的 **“静态三粒度简化版为何略高于复杂自适应门控”、“串行多视角延迟高于单框 `DINOv2`”、“为什么必须多尺度裁剪差分而不能单次裁框输出层特征拼接/剥离”、“`testB/testC` 与 `Zenodo` 定位校准及 `9–13` 类漏映射修复”以及“补充真正零接触最终封口盲测”** 五大关键问题，我们在 `v3_bgopta_and_dtemop/` 完成了严格零泄漏受控实验与方法收敛，确立终稿架构 **[IGD-DCR（输入级隔离多粒度目标—环境差分与定位置信度解耦精化方法）及单趟输出 Token 正交剥离受控探针 BG-OPTA](v3_bgopta_and_dtemop/docs/01_SCI小论文终稿立论_输入隔离差分与输出Token正交探针完整报告.md)**：

1. **核心物理机理突破（为什么必须“输入级空间隔离差分”而非“单框输出层 Token 正交剥离/拼接”？）**：
   - 物理隔离全部测试集、纯源域预注册锁定的单趟 `1.50×` 空间 Token 正交投影探针 **`BG-OPTA`** 证明：虽然输出层 Gram-Schmidt 框外背景子空间剥离无需掩膜即可压制矩形框内 **`67.71%`** 的灰底四角权重（`CitrusUAT` 真值叶片掩膜核验，叶肉能量纯度升至 **`97.39%`**），并在同输入下战胜普通 Token 拼接（`source_dev_bgshift` `+2.67pt`，零丢框 `Zenodo` 2类 `43.38% vs 41.81%` 3/3 种子胜 `Full DINOv2`）；
   - **但在 `SciDB testC`（强反光白板偏移）上，单次扩框 `1.50×` 在进入 `DINOv2` 的 12 层全局自注意力内部已发生前向跨补丁污染（`[CLS]` 暴跌 `-5.12pt` 至 `78.30%`，输出层事后剥离仅能追回至 `79.31%`）**；唯独通过**输入级空间隔离**（`0.82×` 纯叶肉核物理阻断边框反光进入 ViT 自注意力，联合 `2.50×` 环境框做显式特征差分）才能彻底消除污染，达到 **`85.09%`**（高出输出层剥离 **`+5.78pt`**）！
2. **彻底解决速度与简化版质疑（`53.7 FPS` 单批并行模式 + 无参数分歧升级边缘模式）**：
   - 七档零冗余 RTX 4060 Ti 严格 FP32 测速揭示：极小批量下 ViT 受限于 22M 权重显存读取带宽，**单批并行三视角 `IGD-DCR-Batch` 仅需读取一次权重，`testC` 实测延迟仅 `18.61 ms`（`53.7 FPS`，`val` 为 `19.89 ms / 50.3 FPS`）**，比两步串行 `TCD-MGE v2`（`24.13 ms`）快 **`22.9%`**，仅比单框 `Full DINOv2`（`14.09 ms`）多 `4.52 ms`，同时在全部基准取得最高精度（`SciDB val 90.44% / testB 79.28% / testC 85.09%`）；
   - 针对边缘端串行 CPU/NPU，删去冗余熵/马氏超参的**无参数预测分歧升级版 `IGD-DCR-Esc`（仅当 $\hat{y}_{\text{det}}\neq\hat{y}_{\text{std}}$ 时升级）**在 `val` 仅升级 **`3.7%`** 候选框（`testB 22.9% / testC 26.1% / Zenodo 33.2% / SweetOrange 33.5%`），省去 **`66%–96%`** 额外 ViT 算力并保持 **`90.52% / 79.09% / 84.80%`**！
3. **修复 `Zenodo` 14 类零丢框映射 + 真正零接触 `Sweet Orange V1`（`N=1,389`）唯一一次封口盲测全胜**：
   - **修复版零丢框 `Zenodo 11148332`（519 图 / 1,765 框，全 14 类完备映射，0 预测框丢弃）**：`IGD-DCR-Batch` 取得 2类 `AP50 = 45.08±2.15%`、3类 `AP50 = 30.18±1.22%`（较 `Full DINOv2` `+3.27pt / +3.26pt`，**6/6 项逐种子全胜**）；
   - **从未接触过的最终封口盲测集 `Sweet Orange V1`（去重保留原图 `1,389` 张）**：开放 14 类零遮蔽精确命中率从原生 `54.91%`、`Full DINOv2` `56.47%` 提升至 **`IGD-DCR-Esc 60.93%` / `IGD-DCR-Batch 61.17%`（3/3 种子全胜）**，3 症候群 `Macro-F1` 较 `Full DINOv2`（`61.27%`）跃升 **`+11.95pt`** 达 **`73.22%`**！

### `v3_bgopta_and_dtemop/` 核心文件导航
- **终稿论文立论与完整数据报告**：[v3_bgopta_and_dtemop/docs/01_SCI小论文终稿立论_输入隔离差分与输出Token正交探针完整报告.md](v3_bgopta_and_dtemop/docs/01_SCI小论文终稿立论_输入隔离差分与输出Token正交探针完整报告.md)
- **零泄漏预注册与执行记录**：[v3_bgopta_and_dtemop/docs/实验记录.md](v3_bgopta_and_dtemop/docs/实验记录.md)
- **出版级 300 DPI 配图**：
  - [v3_bgopta_and_dtemop/figures/fig1_igd_vs_bgopta_mechanism_and_pareto.png](v3_bgopta_and_dtemop/figures/fig1_igd_vs_bgopta_mechanism_and_pareto.png)
  - [v3_bgopta_and_dtemop/figures/fig2_mask_purity_repaired_zenodo_and_vault.png](v3_bgopta_and_dtemop/figures/fig2_mask_purity_repaired_zenodo_and_vault.png)
- **可复现脚本与全量 JSON 摘要**：`v3_bgopta_and_dtemop/scripts/`、`v3_bgopta_and_dtemop/results/`

---

## 二、阶段性候选与早期探索留档（诚实保留全部演进与纠偏说明）

1. **`v2_tcd_mge/`（三信号流形衰减意见池阶段版，已附规范纠偏说明）**：
   - 在 `SciDB` 与 `Zenodo` 上 3/3 种子战胜 `Full DINOv2`，但存在两步串行调用延迟偏高及早期观察过 `testC` 表现的问题；已明确标定为阶段性方法并校准数据集角色说明（详见 [v2_tcd_mge/docs/01_TCDMGE_SCI小论文定稿与3比3全胜验证报告.md](v2_tcd_mge/docs/01_TCDMGE_SCI小论文定稿与3比3全胜验证报告.md)）。
2. **根级 `docs/`、`scripts/`、`results/`（第一版 `MGS-SymTex` 探索，已降级为历史对照基线 `B2`）**：
   - 经逐种子复核确认 `SymTex` 相对 `Vanilla_KD`、`MGS` 硬覆盖相对 `Full DINOv2` 存在退步种子，如实作为阴性对照留档；全部本地权重 SHA-256 登记于 [MODEL_CHECKPOINTS_MANIFEST.csv](MODEL_CHECKPOINTS_MANIFEST.csv)。
