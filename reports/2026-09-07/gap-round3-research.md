# 第三轮补充调研：五个方法与工程空白

> 日期：2026-09-07
> Agent/任务：五路并行空白调研 / gap-round3
> 对应Issue：#1 的第三轮补充（不改变其 completed 状态）
> 分支：main

## 任务

在已完成的 `reports/2026-09-05/method-landscape-expanded.md`（42 篇、24 个方法家族）、`reports/2026-09-05/literature-followup-v2.md`（18 条、5 个新家族）和 `reports/2026-09-06/kimi-review-supplement.md`（复核纠偏）基础上，识别并填补现有方案中"被点名需要、但没有文献支撑"的五个空白：

1. **弱监督定位**：`阶段性研究思路汇报_20260905.md` 要求比较"分类＋弱定位 vs 器官/病斑检测"，但弱定位一侧零证据。
2. **多轴/缺失标签学习**：Kimi 复核确立的四轴标签体系（器官／症状／病名／证据等级）与"绝不把缺失病名补成推测值"原则，缺可训练的损失方案。
3. **背景鲁棒的可操作技术**：已有 DANN、DG-PLDR 等名词，缺能在 1000 张上直接跑的手段与"问题存在性证明"方法。
4. **可靠性评测协议**：已有共形预测方法侧证据，缺评测指标、协议与基线。
5. **数据线索复查与端侧约束**：HOCD-4 状态待复查；App 端／服务器端推理未定，缺约束数据。

边界：只做调研、核验与分析；不修改代码、中央状态（`PROJECT_STATE.md`）、决策（`DECISIONS.md`）和其他 Agent 报告；不把搜索摘要当最终证据；不把视觉症状当病原学确诊。

## 实际完成

五路并行调研全部完成，共产出 100 余条结构化证据，落盘 `research/inbox/gap-round3-evidence.csv`（99 条）。核心成果按空白分述于"证据与来源"。

一项**必须立即执行的判断修正**（详见下文第六节）：本轮发现 2026 年 Frontiers 已发表跨域植物病害可靠性基准，**"农业缺可靠性评测规范"这一空白主张不再成立**，必须改为引用其作方法学基座，并把可辩护的空白重新收窄。

## 证据与来源

检索范围：2026-09-07，覆盖 CVF open access、arXiv/ar5iv、OpenReview、PMLR、Nature/Springer/Elsevier/Frontiers/MDPI DOI、官方 GitHub、Zenodo/Figshare/Mendeley Data/SciDB/DataCite、Ultralytics 与 Tencent ncnn 官方 benchmark 文档。数据集与仓库状态使用 `curl -sIL`、`git ls-remote`、GitHub API 实际发起请求核验。

### 一、弱监督定位（空白1）

**结论：应放弃把 CAM 弱定位作为定位输出，转而把"点标注"立为正式第三条路线。**

否证"分类＋CAM 弱定位"的三条独立证据：

| 证据 | 数字 |
|---|---|
| 领域内实测（Plant Phenomics 2023，人工逐病斑勾边 GT） | 全监督 DeepLab IoU **0.829**；专门的弱监督 WSLSS **0.434**；CAM+ExG+伪标签 **0.198**；**原始 Grad-CAM 仅 0.107** |
| 方法创新空间（CVPR 2024 "CAM Back Again"） | 大核 CNN ＋**朴素** CAM ＋简单增强，CUB MaxBoxAcc **90.99**，与最新复杂 WSOL 方法持平 |
| 上限与停滞（CVPR 2020／TPAMI 2022 统一协议） | MaxBoxAccV2 三数据集均值：CAM 64.5、HaS 63.9、ACoL 60.9、SPG 63.7、ADL 64.4、CutMix 64.7；"中心高斯"平凡基线 **50.5**；few-shot 全监督 **72.3** |

补充的场景不利证据：Pattern Recognition 2025 指出 CUB 测试集**根本没有小目标**，MaxBoxAccV2／PxAP 系统性掩盖小目标失败——病斑正是小目标，故现有 WSOL 榜单数字对本项目不可外推。WACV 2025 进一步指出现有 WSOL 数字普遍依赖 GT bbox 调阈值与选模型（变相偷用全监督），只用类别标签选模型时定位性能显著下降。柑橘域内亦有定性证据（Computers 2025）：DenseNet121／ResNet50 的激活"并非都对应主要病斑"、"部分落在无害背景"。

点标注路线的成本与性能证据：

| 标注形式 | 整图耗时（VOC 实测） | 单目标耗时（COCO 实测） |
|---|---|---|
| 图像级 | 18.5 s | — |
| **单点** | **22.1 s（仅贵 1.19 倍）** | **0.9 s** |
| 涂鸦 | 34.9 s | — |
| 框（extreme clicking） | — | 7 s（传统拖拽画框 25.5 s ＋ 校验 9.0 s） |
| 全掩膜 | 239.7 s | 79.2 s |

- 单点→框：P2BNet（ECCV 2022）COCO 22.1 AP／47.3 AP50；P2Object（IJCV 2025）25.9 AP／49.0 AP50，而全框 Faster R-CNN 上界 37.4／58.1，即达**约 70% AP、84% AP50**。
- 少量全标注＋大量点标注（与本项目"专家复核＋农民初标"同构）：Point-DETR（CVPR 2021）在 COCO 20% 全标注下 33.3 AP；Group R-CNN（CVPR 2022）在 5% 全标注设定下超 Point-DETR 3.9 mAP；Point-Teaching（AAAI 2023）在 5% 下再超 Group R-CNN 3.1 AP。
- 单点→掩膜：Pointly-Supervised Instance Segmentation（CVPR 2022）框内 10 点达全掩膜监督 **94%–98%**，已集成进 Detectron2。
- 农业内实证：PIS-Net（AIA 2024）水稻田杂草每实例约 13 点达全监督约 **90%**；BMC Plant Methods 2020 点标注＋主动学习高粱省 >55%、小麦省 >50% 标注时间——但该文明确发现**纯点监督只能得弱学习器**，最终改用点标注筛选待全标注样本。

一个值得验证的反直觉假设：Plant Phenomics 2023 中弱监督 WSLSS 跨域（苹果→葡萄／草莓）IoU **0.511 反超**全监督 DeepLab 的 **0.458**。该观察 n=1 不足采信，但与本项目 Mendeley→CitrusUAT 掉点痛点直接相关，若成立则本身是非拼接式发现。

### 二、多轴/缺失标签学习（空白2）

**结论：四轴标签体系可完全用损失层面改动实现，零额外可训练参数。**

"未观测≠负例"的代价已被量化：Cole et al.（CVPR 2021）COCO 微调下 BCE 全标签 75.8，而把未观测当负例（AN）仅 **64.1**，差 **11.7 mAP**；且只换损失即可挽回大部分差距。这为项目"绝不把缺失病名补成推测值"提供了直接文献支撑。

推荐配方（总损失三项由 Kendall uncertainty weighting 自动加权）：

| 轴 | 方案 | 关键依据 |
|---|---|---|
| 器官（不缺失） | 标准 softmax ＋ CE，作层级锚点 | — |
| 症状（多选、可缺失） | **Hill loss**（负例项 `L⁻=−(λ−p)p²`，λ=1.5）＋ **partial-BCE 归一化**（`g(p)=αp^γ+β`，α=−4.45, β=5.45, γ=1）；第 2 epoch 后可选启用 SPLC（τ=0.6）；保守替代为 **LL-R**（只丢弃大损失项不翻转，Δ_rel=0.2） | COCO 单正标签 BCE 68.57 → Hill 73.17 / SPLC 73.18；COCO 10% 标签 BCE 69.91 → partial-BCE 72.15 |
| 病名（最深、可缺失） | **conditional softmax**：只在兄弟节点集内归一化，仅在父标签为正的样本上计算子分类器损失；＋ **logit adjustment**；＋ **marginal loss** 思想（未观测病名集合合并为虚拟类，概率取边缘和） | CoNLL 2024：HWV 上 Cond.Softmax+LA Macro-F1 **54.39** 反超专门层级 SOTA HITIN 的 51.73；MedIA 2020 提供边缘概率处理部分标注的原始论证 |
| 证据等级（不缺失） | **不作预测目标**，而作损失权重与结构：样本权重 → **GLC 前向修正**（PCR／专家为 trusted set 估计转移矩阵）→ 两阶段训练（先全弱标签，再冻结骨干在专家＋PCR 子集微调） | GLC（NeurIPS 2018）MNIST 5% 可信数据下 61.12 → **95.45** |

**黄龙病硬约束的数学实现**：黄龙病节点损失**只在证据等级为 PCR／实验确证的样本上计算**，其余样本对该节点掩码；视觉预标的"疑似黄龙病"进入**症状轴的"斑驳"＋"黄化"**而非病名轴。这把项目红线直接编译为一个掩码条件，无需额外机制。

评测：主指标须在"全轴完整标注＋证据等级≥专家确认"的黄金测试集上计算；每轴分开报告（器官 accuracy／症状 per-class AP／病名 **hF1**）；弱标签测试集数字只作附属对照并标注不可与文献比较。理由：arXiv 2109.11282 正式证明非可分解指标（mAP／F1）的无偏估计量方差显著增大、可能导致病态优化；arXiv 2207.13186 亦承认领域内尚无专门的无偏 mAP 估计量，并主张配一个小的偏置受控留出集。

一个高度相关的先例（待核实 DOI 与数据可得性）：Qiu et al. 2022（Frontiers in Plant Science）柑橘 HLB 症状级标注数据集，7,516 张无增强图，其分组为"健康／典型 HLB／**疑似** HLB／**类** HLB／HLB 无关"——**本质即一个证据等级轴**，且症状类别中的煤污、溃疡与本项目症状轴重合，标注形式为症状级 bbox。

### 三、背景鲁棒的可操作技术（空白3）

**结论：有现成、可零标注成本复刻的量化诊断协议；同时一整条"因果/不变表征"路线应被排除。**

诊断侧（论文"问题存在性证明"章节的核心）：
- **IN-9 八变体 ＋ BG-Gap 单标量指标**（ICLR 2021，`MadryLab/backgrounds_challenge` 开源）：ResNet-50 在 IN-9L 上 Original 96.3／Mixed-Same 89.9／Mixed-Rand 75.6／Only-BG-T 43.6／Only-FG 84.1，**BG-Gap 14.3%**；对抗背景可误导 **87.5%** 的正确前景。**本项目已有 CitrusUAT 的逐图二值掩膜，可零额外标注成本复刻整套协议**（掩膜标注对象是"病斑"还是"整叶"需先下载核实，见"未完成或阻塞"）。
- **8 像素探针**（arXiv 2206.04374，开源）：PlantVillage 上仅取四角＋四边中点共 8 个背景像素训随机森林，测试准确率 **49.0%**（38 类，随机基线 2.6%）。一小时内可完成，是最有冲击力的单一数字。
- **关键警告**：同文用模糊度做预测器，证明仅前景版本偏差量（10.0%）与含背景版本（11.7%）**几乎相同**——主因是**采集偏差（capture bias）**而非语义背景。故单靠分割前景不能解决问题，必须叠加光度类强增强。

合成侧的正反证据：
- 农业内正面数字（Ecological Informatics 2025 FBR）：加背景重组各作物 **+9.9~23.6%**，葡萄 57.4%→81.0%，CDAN+E＋FBR 真实设定 **91.1%** vs 分布对齐类 57.7%。
- 同文诚实负面数字：**FBR 单用在苹果真实设定仅 57.0%**（Real-to-Real 上限 97.7%），作者自归因于合成伪影与不自然边界。
- 跨任务陷阱：CVPR 2021 在实例分割上"随机粘贴就够"且 blending 消融显示有无高斯边缘平滑结果相同（50.0/43.7）；但 ECCV 2018 在**目标检测**上随机放置仅 **56.9% mAP** vs 上下文引导 **64.0%**，原文称随机放置"根本不工作、甚至可能有害"。**本项目是分类/检测＋1000 张小数据，应按后者预期**。
- 防伪影捷径的标准做法（ICCV 2017）：同一合成图生成多种 blending 变体，使边界线索失去判别性。

低成本正选：
- **TrivialAugmentWide**（ICCV 2021 Oral，已入 torchvision）：完全无参、零搜索、零显存。
- **DFR 末层重训**（ICLR 2023 Oral）：ERM 模型已学到 core features，仅重训末层线性层即匹配或超过 Group DRO，成本低几个数量级，单卡数分钟。
- **LFLSeg**（IEEE T-ASE 2022，LeafGAN 的组件）：无掩膜标注的弱监督前景分割器（全叶／部分叶／非叶三分类，验证 99.8%，全叶 F1 83.9%），可替代 SAM 做前景先验。同文对照极有说服力：vanilla CycleGAN 仅 **+0.7%**，而保留原背景的 LeafGAN **+7.4%**。
- 多源配方排序参考（arXiv 2510.12909 TMPS，每病害仅 10 张目标域样本）：**混合训练 < 微调 < 目标感知度量学习＋优先采样**（macro-F1 分别 +7.3／+3.6 优势）。

一个真实的方法空白：农业领域**没有**在同等预算下把"强增强"与"复杂 UDA"做公平对照的基线研究。若本项目跑出"强增强 ≥ DANN"，本身即可发表的负面结果。

### 四、可靠性评测协议（空白4）

**结论：现有主流指标（ECE、AURC、E-AURC）在本项目数据规模下均不可作主指标，须整体替换。**

| 指标 | 判决 | 依据 |
|---|---|---|
| **ECE（15 等宽箱）** | 降级为诊断，不作主指标 | AISTATS 2022：**检测 2% 误校准需 >10,000 样本**，且 n=200 时**完美校准的模型**会被系统性判为大误差；STOC 2023：ECE 不是 consistent calibration measure（不连续反例）；NeurIPS 2019：对 scaling 类方法其校准误差**本质不可测**；CVPRW 2019：重校准方法排名随箱数／norm 剧变 |
| **主指标改用 Brier／NLL** | 采纳 | NeurIPS 2022：校准误差估计量"通常有偏且不一致"；proper score 无偏且无分箱超参 |
| **classwise-ECE／class-j-ECE** | 黄龙病类必须单独报 | NeurIPS 2019 Dirichlet calibration；聚合 ECE 会被健康与常见病害类淹没 |
| **E-AURC** | **排除** | ICLR 2023 反例：两个排序能力完全相同（均无区分力）的模型，准确率 20% vs 80%，E-AURC 会偏好后者——未能如设计意图剥离分类器性能 |
| **AURC 作主排序** | **排除** | NeurIPS 2024：AURC 与 E-AURC **违反单调性**；换用 AUGRC 后 **6 个数据集中 5 个的 Top-3 排名改变** |
| **AUGRC** | **选择性预测主指标** | NeurIPS 2024，闭式分解 `AUGRC=(1−AUROC_f)·acc·(1−acc)+½(1−acc)²`，把"分类器多准"与"置信度排序多好"显式解耦 |
| **OSCR** | 开放集主指标 | NeurIPS 2018：CCR 要求**分类正确**而非仅未拒识——本项目"把缺素拒掉但把真黄龙病判成溃疡病"同样是错的，AUROC 看不出这一层 |
| **AUROC（OOD）** | 必配双向 AUPR 且报 Base 值 | ICLR 2017 原文：AUROC 在正负类基率差异大时不理想；同一模型 AUPR-Succ 100 而 AUPR-Err 仅 **48**（Base 1.7） |
| **openness** | 弃用 | ICLR 2022：语义相似度的影响是类别数量的约 **10 倍**（5–6% vs 0.6%） |
| **D-ECE**（若做检测） | K≤2、每维箱数≤5 | CVPRW 2020：`N_total=ΠN_k`，K=3 每维 10 箱即 1000 箱，小数据不可用；反例显示标准 ECE 与 D-ECE 给出**相反排序** |

必须包含的强基线（四篇 ICLR/NeurIPS 一致结论）：
- **训练充分的 MSP**：ICLR 2023 Oral（FD-Shifts）实测优于 ConfidNet、Deep Gamblers、SelectiveNet、DUQ 与 deep ensembles。
- **训练充分的 MLS（max logit）**：ICLR 2022 Oral 实测平均 **+15.6% AUROC**；TinyImageNet 上沿用多年的 MSP 基线 57.7，仅改训练配方→82.7，再换 MLS→**83.0，超过当时 SOTA ARPL+CS 的 78.2**。闭集准确率与开放集 AUROC **Pearson ρ=0.95**。
- **投入优先级应反转**：与其实现复杂开放集专用方法，不如先把闭集准确率榨干（更强骨干、AugMix/RandAugment、更长调度）。OpenOOD v1.5 进一步证明数据增强与 post-processor **效果互补**（AugMix+ASH=82.16% 为其全场最佳）。
- **排除 SelectiveNet／Deep Gamblers**：把其选择机制换成朴素 softmax response 后性能一致变好（ImageNet100@80%cov：6.00→4.47；5.21→4.52）；且 SelectiveNet 每个目标 coverage 需单训一个模型。
- **慎用 Mahalanobis**：ICML 2021 Workshop 实测其在全数据下胜 ODIN 12/16，**低数据仅 3/16**——本项目每类仅几十张，类条件协方差估计必然病态，改用 KNN。

共形预测的样本量红线（α=0.1，`Cov~Beta(n+1−l, l)`，`l=⌊(n+1)α⌋`）：

| n | 均值 | 标准差 | 95% 区间 |
|---|---|---|---|
| 100 | 0.9010 | 0.0296 | [0.836, 0.951] |
| **200** | 0.9005 | 0.0211 | **[0.856, 0.938]（±4pp）** |
| 500 | 0.9002 | 0.0134 | [0.873, 0.925] |
| 1000 | 0.9001 | 0.0095 | [0.881, 0.918] |

Angelopoulos & Bates 的 Table 1 给出：达到 ±0.05 需 n=102，±0.01 需 **n=2491**。**留 200 张校准集时"90% 覆盖保证"实际波动 ±4pp，几乎抹掉其实用意义——这量化支持了项目"共形预测只在普通校准不足时加入"的既定决策，建议把此表直接放进汇报作为该决策依据。**

分组划分的效应量硬数字：ISPRS Open J. 2022 遥感分割实测**随机划分比空间块划分高估最多 28%，且训练样本越小偏差越严重**；同一次采集内部自相关显著高于跨采集（可直接类比连拍组）。Data in Brief 2026 给了植物病害领域的直接去重审计模板：甘蔗叶数据集 6748 张→去精确重复 5695→去近重复 4348，**71 个 pHash 组／178 张横跨划分边界**，macro-F1 随机划分 0.9774 vs 分组划分 0.9645。

OOD 干扰测试的协议要点：
- 理论背书（AAAI 2020）：不能拿随机的其它数据集当 OOD（那属 far-OOD，太容易）；本项目用"缺素／日灼／药害"这类同域、同物种、症状相似的非生物胁迫作未知类，**恰是该文主张的"语义性质的 out-distribution"**。
- 划分硬约束（OpenOOD v1.5）：不是样本不重叠，而是**类别标签空间不相交** `Y_OOD^val ∩ Y_OOD^test = ∅`；须**按胁迫类型划分**（如 val=缺镁/缺锌，test=缺铁/缺锰＋日灼＋药害），按图片随机划是变相泄漏。
- **已知未知 vs 未知未知分开报**（NeurIPS 2018 的 `B`/`A` 框架）：缺素（可采集可标注）属 `B`，拒识率是**乐观上界**；日灼、药害等属 `A`，拒识率才是**真实泛化能力**。
- **full-spectrum 组（田间刚需）**：阴天／逆光／雨后／不同物候期的健康叶与已知病害叶**仍算 ID**，不能因拍摄条件变化被拒识。OpenOOD v1.5 实测这会让 near-OOD AUROC 掉 >10%；不设这组，模型上线会因"阴天全部拒识"而不可用。
- **数据纯净性风险**（ICML 2023 NINCO）：某些常用 OOD 集 **>50% 样本实含 ID 物体**。本项目对应风险是**缺素样本里混有实际感染 HLB 的植株**（缺素与 HLB 可同株共发，且 HLB 本身表现为缺锌样斑驳黄化），而硬约束是无 PCR 不能确诊，**无法靠肉眼保证未知集纯净**。做法：缺素样本做 PCR 阴性确认或至少来自无病史田块；逐样本人工审查并记录流程；无法全部 PCR 时**如实报告污染率上界**。可用 UF/IFAS EDIS PP328 的 "pen test"（HLB 斑驳黄化跨主脉不对称、两侧不一致；缺素两侧对称一致）作标注规程依据。
- **合理预期**：near-OOD 现实区间 **75–85 AUROC**（OpenOOD v1.5：near 78.17 vs far 95.74，差 18 点；ICLR 2022：CUB Hard OSCR 73.1 而闭集准确率 86.2，gap 13 点）。**不要拿文献里 90+ 的 far-OOD 数字做目标或对外承诺。**

支撑本项目问题困难性与临床意义的柑橘域数字：
- Plants 2023（金标准 qPCR，6 类）：总体 96.92%、HLB 阴阳性 98.08%，但**缺锌 vs 缺锌＋HLB 阳性仅 92.86%**——证明"缺素×HLB"是最难一档。
- Frontiers 2022（12 类）：**熟练人员判定的斑驳黄化，PCR 阳性率仅 86.49%**——视觉标签≠病原学确诊的量化证据。
- Frontiers 2021：已有"**疑似 HLB**"类先例（PCR ＋ 矿质分析双确认），可对标。
- Plants 2025：**仅用专家目视标签**却报 RF F1 99.8%，作者自陈"应做 PCR"——正是本项目要批判的对象。

### 五、数据线索复查与端侧约束（空白5）

**HOCD-4 复查结论：彻底不可用，建议从方案中永久移除、不预留位置。** 实测命令与输出：

```text
$ curl -sIL -o /dev/null -w "%{http_code}\n" https://github.com/AlanPikaso/HOCD-4
404

$ git ls-remote https://github.com/AlanPikaso/HOCD-4.git
remote: Repository not found.
fatal: 仓库 'https://github.com/AlanPikaso/HOCD-4.git/' 未找到

$ curl -s https://api.github.com/repos/AlanPikaso/HOCD-4
{"message": "Not Found", "status": "404"}

$ curl -s https://api.github.com/users/AlanPikaso | grep -E 'public_repos|created_at'
  "public_repos": 0,
  "created_at": "2024-01-04T13:14:50Z",

$ curl -s "https://api.github.com/users/AlanPikaso/repos?per_page=100"
（HTTP 200，返回空数组）
```

账号存在但**公开仓库数为 0**，仓库列表为空数组——不是改名或迁移，而是作者把所有公开内容撤下。补充核验：GitHub 仓库搜索、Zenodo／Figshare／Mendeley／DataCite 全库检索、Web 搜索均无备份或二次索引。唯一路径是邮件联系通讯作者。

**一个对立项有正面价值的空白确认**：全库检索（Zenodo／DataCite／Mendeley）下，柚类（*Citrus grandis* / *maxima*）病虫害图像数据**仅有一个条目**（Mendeley 10.17632/vk7dnf8nrj.2），且其 v1 的 868 张与项目已有"柚类 868 张真实原图"**同源重复**，v2 增量全是增广。即**柚类病虫害图像是全球性数据空白，本项目 1000 张自采数据本身具有发布价值**。

本轮新增且实测可下载的高价值候选（虫害线索质量高于病害线索）：

| 数据集 | 规模 | 标注 | 实测状态 | 定位 |
|---|---|---|---|---|
| BDLemonLeaf v6（10.17632/643f5bbc2t.6） | 3431 张／13 类 | **bounding box**（格式未标明） | Root.zip 96.2MB，跟随重定向后 **206 ＋ PK 头验证** | 虫害/病害源域预训练（潜叶蛾、锈壁虱、红蜘蛛） |
| 孟加拉大规模柠檬集（10.17632/8d9fv6kpt3.1） | 17609 张／18 类 | 分类（文件夹） | 落地页 200，文件 API 返回空/504 | 虫害源域（**煤污病 1597、潜叶蛾 900**，正命中症状轴） |
| CitrusFic 葡萄柚锈壁虱（10.5281/zenodo.21250372） | 3154 张 | 分类 | zip 2.49GB，range GET **206 ＋ PK 头** | 柚类近亲虫害源域 |
| CTSD-YOLO-Split（10.57760/sciencedb.40726） | 3.44 GB | **YOLO 框＋分割掩膜** | 落地页 200，文件清单未取到 | 果园背景/遮挡先验 |

最贴合本项目场景但需邮件申请：**YOLO-Citrus**（Frontiers 2025，华南农大，1308 原图，含溃疡/HLB/锈病/沙皮病，**果园 30–100cm 近距**，Canon R8 ＋ iPhone 15 Plus 采集）——采集距离与相机组合与本项目 App 方式高度一致，Data Availability 为"应作者要求提供"，无 DOI 无链接。

端侧与服务器端决策表（条件化，不锁死）：

| 情形 | 模型规模上限 | 分辨率 | 可选架构 | 对研究方案的约束 |
|---|---|---|---|---|
| A. 纯 App 端 | 检测 ≤3M params／≤10 GFLOPs；分类 ≤10M | 检测 640／分类 224 | YOLO11n/YOLO26n；RepViT-M1.0 或 FastViT-T12 | **DINOv2、RT-DETR 全系排除**。ncnn 官方 benchmark 中 `vision_transformer` 在 Pi5 上 **611.65ms**；RT-DETR-R18 约 60 GFLOPs 是 YOLO11n 的 9 倍，且 deformable attention 在 NCNN/TFLite 算子支持存疑 |
| B. 纯服务器端 | 检测 ≤76M；表征 ≤300M | 检测 640／细粒度 448–512 | RT-DETR-R18/R50、YOLO11m/l、DINOv2 ViT-S/B ＋ LoRA | 可用，但**训练侧受 4060 Ti 16GB 限制**：DINOv2 ViT-B/14 @512 batch8 需 **19GB 会 OOM**（官方 issue #553），必须降到 224 或用 LoRA |
| **C. 混合（推荐）** | 端侧 ≤3M 检测器；云端不设上限 | 端侧 640 检 ROI／云端 448 细分类 | 端侧 YOLO11n（NCNN）＋置信度门控；云端 DINOv2+LoRA 或 RT-DETR | **当前应采纳**：训练阶段照常用重模型做上界与蒸馏教师，部署阶段再按最终决策裁剪。端侧 NCNN 是 ARM 最快路径——Pi5 上 **67.03ms vs PyTorch 299.09ms 且 mAP 反升**（0.4784 vs 0.4760） |

量化与转换的关键约束：

| 来源 | 场景 | FP32→INT8 | 掉点 |
|---|---|---|---|
| Ultralytics Jetson 官方 | COCO 检测 nano，TensorRT | 0.4790→0.4490 | **−3.0** |
| Ultralytics OpenVINO 官方 | COCO 检测 nano | 0.4762→0.4634 | −1.3 |
| arXiv 2508.19600 | YOLO12n 静态 INT8 | 0.4047→0.3325 | **−7.2** |
| 同上 | YOLO12x（大模型对照） | 0.5514→0.5202 | −3.1 |

三条机制性结论：① **模型越小掉点越大**（同一实验 n 掉 7.2、x 只掉 3.1），端侧必用 nano 级，恰在最脆弱区间；② **FP16 与动态 w8a32 几乎零损失，只有静态 INT8 掉点**——锅在校准质量而非 8-bit 本身，故优先 FP16／动态量化；③ 失败的直接机制是激活值动态范围——TF 官方记录 EfficientNet PTQ 时 top-1 从 75% 崩到 46%，定位到某张量范围 −168~204 塞不进 int8 桶，**把 swish 换 relu6 后恢复 74.4%**；**YOLOv8/11 全系用 SiLU（swish 家族），PTQ 风险显著高于 relu6 骨干**。

**一个必须自测、不得引用他人的空白**：检索 arXiv 2508.19600 全文、AIMET 白皮书、EasyQuant 等，**没有任何论文或官方文档给出 INT8 vs FP32 的 AP_small 分解**，全部只报聚合 mAP50-95。柑橘病斑是小目标＋细粒度纹理，聚合 mAP 会系统性掩盖损失。

转换链路的坑：**PyTorch→ONNX→NCNN 是错误路径**，腾讯官方推荐 PyTorch→**PNNX**→ncnn，`onnx2ncnn` 已是 legacy 并在运行时自行提示改用 PNNX；ONNX→TFLite 的 NCHW/NHWC 是结构性问题（onnx2tf 作者在 README 明说转换"必然在某处崩掉"）；CoreML 内嵌 NMS 只支持 `dynamic=False`；ExecuTorch 只支持 FP32 导出。**INT8 不等于提速**：ncnn 官方表中 RK3588 上 resnet18 FP32 18.86ms → INT8 **34.00ms**（慢近一倍），vgg16 107→283.94ms，收益强依赖 SoC 是否有 armv8.2 dot product 指令。

### 六、必须修正的既有判断（本轮最重要的单条结论）

**"农业领域缺可靠性评测规范"这一空白主张已不成立。**

Xiang et al. 2026（Frontiers in Plant Science，DOI 10.3389/fpls.2026.1826962；PMC13236948）已完成 PlantVillage→PlantDoc 的"校准＋选择性预测＋开放集"三合一基准：21 共享类 ＋ 490 未知类，5 个 backbone，**10 次 parent-image-aware 分组划分（seed 42–51）＋95% CI ＋ 配对 Wilcoxon**。其关键数字：准确率 99.73%→**32.10%（掉 67.7pp）但均值置信度仍 79.76%**；ECE 0.4761→温度缩放后 0.3645（T=1.54）；risk@80%coverage = 0.6430；固定风险 5–15% 时 coverage 崩到 **0.0029**；**Setting A（源域标阈值）名义 80% 实际只答 5.1%**（阈值不跨域迁移的实证）；能量分数 AUROC 0.5832–0.6101，DINOv2 最好 0.7637。

**若按原设想设计，会被判为重复工作。** 应主动引用其作方法学基座，并把可辩护的空白按强度重排：

1. **HLB 在"PCR 不可得／不完美参考标准"下的可靠性评测**——本轮覆盖的 7 篇 HLB 视觉论文**无一报告 ECE／OOD AUROC／risk-coverage**。
2. **把缺素等非生物胁迫设为需拒识的 near-OOD 并报 OSCR／AUROC**——现有农业开放集协议的未知类**全是留出的病害类别，无人做生理性黄化**。
3. 植物病害上的类条件共形预测。
4. 农业上的训练式弃答器／learning-to-defer（现有全是 post-hoc 阈值）。

①②是强主张，且都有柑橘域数字（缺锌＋HLB 仅 92.86%、斑驳黄化 PCR 阳性率仅 86.49%）支撑其困难性与临床意义。

另有两个 venue／文献更正，供后续引用时避免出错：
- OpenOOD **v1.5 是 DMLR 论文，不是 NeurIPS**（只有 v1 是 NeurIPS 2022 D&B）。
- Neal et al.（ECCV 2018）**没有 arXiv 预印本**，必须引 CVF open access 或 DOI 10.1007/978-3-030-01231-1_38。
- "Bayesian Confidence Calibration for Epistemic Uncertainty Modelling" 是 **IV 2021** 而非 ICRA 2021；Miller et al. "Dropout Sampling for Robust Object Detection in Open-Set Conditions" 是 **ICRA 2018** 而非 2019。
- 两条**不存在或被误记的文献，勿引用**：Barbedo "Plant disease detection: a critical review"（该确切标题检索范围内不存在；真正的批判性综述是 Barbedo, *A review on the main challenges in automatic plant disease identification based on visible range images*, Biosystems Engineering 144:52-60, 2016）；"Thapa 的方法学批判文献"（Thapa 是 PlantPathology 2020 / FGVC 数据集论文作者，非方法学批判）。

## 新增发现与排除项

### 新增候选（按优先级）

| 优先级 | 空白 | 候选 | 与现有方案的关系 |
|---|---|---|---|
| A | 弱监督定位 | **点标注弱半监督检测**（Group R-CNN / Point-Teaching，Faster R-CNN 基座） | 立为"分类＋弱定位 vs 器官检测"之外的**第三条路线**，且成本最优 |
| A | 缺失标签 | Hill/LL-R ＋ conditional softmax ＋ marginal loss ＋ GLC | 四轴标签体系的可训练实现，零额外参数 |
| A | 背景鲁棒 | IN-9 八变体＋BG-Gap＋8 像素探针 | 论文"问题存在性证明"章节 |
| A | 背景鲁棒 | TrivialAugmentWide ＋ DFR 末层重训 ＋ 先源后目标微调 | 替换 IRM/Group DRO/复杂 DG 路线 |
| A | 可靠性评测 | AUGRC（主）＋ OSCR ＋ SAC ＋ Brier/NLL ＋ classwise-ECE | 整体替换 ECE/AURC/E-AURC |
| A | 可靠性评测 | 训练充分的 MSP／MLS 作强基线 | 必须包含，否则会被直接质疑 |
| A | 数据 | BDLemonLeaf v6、孟加拉柠檬集、CitrusFic | 虫害源域预训练 |
| A（流程） | 数据 | 分组划分双报告（随机 vs 分组）＋ pHash 去重审计 | 泄漏量作为结果之一报告 |
| B | 背景鲁棒 | LFLSeg（无掩膜标注的前景分割器） | 替代 SAM 做前景先验，绕开 SAM 在繁茂背景失效 |
| B | 数据 | CTSD-YOLO-Split、CitDet | 果园背景与遮挡先验 |
| B（申请） | 数据 | YOLO-Citrus（华南农大） | 采集方式最贴合，需邮件申请 |

### 排除项与理由

| 排除项 | 理由 |
|---|---|
| 自研 CAM 变体 | CVPR 2024 证明大核骨干＋朴素 CAM 即达 SOTA，所有变体收益可被骨干替换抹平；落入"无依据拼接创新" |
| ADL/HaS/ACoL/SPG/CutMix 等经典 WSOL | 统一协议下三数据集均值全部不超过 CAM（最大差 +0.2） |
| CAM 弱定位作为**定位输出** | 领域内实测 Grad-CAM 病斑 IoU 仅 0.107；仅可保留为可解释性审计工具，且须先跑 sanity check |
| 纯图像级 WSOD（WSDDN/OICR/C-MIL/CASD/MIST 谱系） | VOC2007 从 2020 的 56.8 到 2024 的 58.3，四年 1.5 点，明确饱和；失效模式（只框最判别部件）恰是病斑典型形态 |
| SAM 直接做果园前景分割 / Pro2SAM 零样本病斑分割 | IEEE Access 2024 作者自述失败模式：背景过于繁茂时绿色背景块被误判为叶片；EMSAM 2025 显示 SAM 类在**轻病害/小病斑**上被 CNN 明显反超（Swin-Unet IoU 仅 0.3195） |
| **IRM / IRMv1 / 因果不变表征整体** | 线性设定要求**环境数 E > d_e 且此为必要条件**（本项目源域 2–4 个）；IRMv1 在 IRM 自身玩具问题上即失效；**更致命**：本项目背景变化属**协变量偏移**，按 ICLR 2021 sample-complexity 一文自己的理论，此设定下 IRM 与 ERM 渐近同解 |
| 复杂 DG 方法堆叠（13 种） | DomainBed：统一条件下**无一超过调优 ERM 1 个点以上** |
| 朴素 Group DRO | 过参数化下平均训练损失趋零即最差组趋零，朴素形式数学上无效；且 DFR 以低几个数量级成本匹配 |
| CycleGAN 类风格迁移做背景域适应 | 黄瓜五类病害上 vanilla CycleGAN 仅 **+0.7%**，而保留背景的 LeafGAN 为 +7.4% |
| FBR／背景重组**单独使用** | 苹果真实设定单用仅 57.0%（上限 97.7%），作者自归因合成伪影 |
| "随机位置 copy-paste 就够" | 只在实例分割成立；目标检测上随机 56.9% vs 上下文引导 64.0% |
| 粘贴伪标签物体 | CVPR 2021 明确报告无额外 AP 增益 |
| **ECE（15 等宽箱）作主指标** | 六条独立缺陷（分箱有偏、需 >10,000 样本、非 consistent measure、无类条件性、排名随箱数剧变、对 scaling 类不可测） |
| **E-AURC / AURC 作主排序** | 反例证明 E-AURC 仍与准确率相关；二者违反单调性，换指标后 5/6 数据集 Top-3 排名改变 |
| SelectiveNet / Deep Gamblers | 换成朴素 softmax response 后性能一致变好；SelectiveNet 每 coverage 需单训一模型 |
| ConfidNet / DUQ / MC-dropout 作选择函数 | FD-Shifts 实测朴素 MSP 优于它们；Sci Rep 2017 发现 standard dropout **比随机转诊还差** |
| 专门开放集方法（OSRCI/OpenHybrid/ARPL+CS） | 训练充分的 MLS 基线即超过当时 SOTA；此前四年增益**主要来自基线训练不足** |
| openness 作难度度量 | 影响力仅为语义相似度的 1/10 |
| Mahalanobis（本项目条件下） | 低数据下从胜 ODIN 12/16 跌到 3/16；改用 KNN |
| ASH / ReAct 用于 transformer 骨干 | 为 CNN 定制，在 transformer 上显著退化；应改用 RMDS |
| PDQ / MUE 作检测可靠性主指标 | PDQ 需像素级分割标注＋角点协方差；MUE 对不确定性尺度不敏感（同乘/同加常数不变） |
| vector/matrix scaling、PTS、spline 校准 | 参数量随类别数增长，小数据必崩；只用单参数温度缩放 |
| AN-LS（label smoothing） | 均匀基准下优于 AN，但**标注偏置下最脆弱**（平均降 5.1 mAP，size bias 下 64.8→56.7 甚至低于 AN 的 57.0）；农民标注必然是显著性偏置 |
| ROLE / 在线标签估计 | 需为每样本维护 N×C 可学习标签向量；在 CUB 细粒度小数据上表现最差（15.0，低于 AN 的 19.1） |
| PCGrad / GradNorm | 掩码训练下各轴梯度来自**不相交样本子集**，冲突检测与相对训练速率的假设不成立（此为推断，文献未直接讨论该失效模式）；先只用 uncertainty weighting |
| HXE / soft hierarchical labels | iNat'19 上 top-1 error 从 44.04 升到 52.06，牺牲 8 点换"错得更体面"不划算 |
| 无偏 mAP / propensity-scored 指标 | 非可分解指标的无偏估计量方差爆炸、可能病态优化；本项目 propensity 未知且仅 1000 张；改用黄金测试集 |
| Tanno et al. 迹正则（原始形式）/ AggNet / EM 众包聚合 | 均需每样本多标注者重叠；且 EM 恢复的"共识"会继承农民与专家的共享偏差，GLC 的 trusted-set 锚定才是正确框架 |
| Point-DETR 作**实现选型** | 设定证据保留，但 DETR 基座在约 1000 张小数据上收敛困难；改用 Group R-CNN / Point-Teaching |
| HOCD-4 | 仓库 404、作者 public_repos=0、无任何备份或镜像 |
| AgriLeafNet（fkkfx4vdm3） | 标准化白纸 A4 背景，与"复杂果园背景"核心诉求完全相反 |
| IPTIS 柚树集 / NeSTCount / Zenodo UAV 橙果 | UAV 尺度或纯计数任务，与近距叶部病斑不匹配 |
| INRAE 054NAB 多光谱 | 14 波段、单文件约 38GB；App 只采 RGB；且标签为视觉判定非 PCR |
| Citrus maxima 柚集 v2 增量 | v1 的 868 张已在项目清单内，v2 新增全是增广 |
| MobileViT | 同精度下比 RepViT/FastViT 慢 5–7 倍 |
| DINOv2 / RT-DETR 用于**端侧部署** | ncnn 上 ViT 类在 Pi5 需 611ms；RT-DETR-R18 是 YOLO11n 的 9 倍 FLOPs。训练侧仍可用作上界与蒸馏教师 |
| "农业缺可靠性评测规范"作为空白主张 | 已被 Frontiers 2026（PMC13236948）填补；应改为引用其作方法学基座 |
| 仅用专家目视标签并报高分 | Plants 2025 的 RF F1 99.8% 建立在纯目视标签上（作者自陈应做 PCR）；Frontiers 2022 实测熟练人员判定斑驳黄化 PCR 阳性率仅 86.49% |

## 修改文件

- 新增 `reports/2026-09-07/gap-round3-research.md`（本文件）
- 新增 `research/inbox/gap-round3-evidence.csv`（99 条结构化证据）

未修改 `coordination/PROJECT_STATE.md`、`coordination/DECISIONS.md`、`coordination/CURRENT_ROUND.md` 及任何其他 Agent 的报告。

## 验证命令与结果

```bash
$ python scripts/validate_repository.py
仓库校验通过。

$ git status --short
?? reports/2026-09-07/
?? research/inbox/gap-round3-evidence.csv

$ wc -l research/inbox/gap-round3-evidence.csv
100 research/inbox/gap-round3-evidence.csv
```

数据集与仓库状态核验命令及输出见"证据与来源"第五节（HOCD-4 复查部分为完整实测记录）。

## 未完成或阻塞

**必须核原文才能引用的（当前仅二手转述或被付费墙阻挡）**

1. **FBR 论文**（Ecological Informatics 2025）逐作物×逐 UDA 方法完整消融表在付费墙后，未找到代码发布。这是本项目最直接的对标基线，建议通过机构渠道取全文。
2. **Gui et al. 2021**（Comput. Electron. Agric. 191:106523）常被引用的 99.72%→41.81% 降幅来自后续文献转引，未从一手全文核实；逐组件消融表同样在付费墙后。
3. **TMPS**（arXiv 2510.12909）仅取到摘要级数字，未取到逐作物明细，代码是否公开未说明。其"混合训练 < 微调"排序对多源配方设计很关键。
4. **HLB 人工视觉诊断上限 ≤59%**（Futch et al. 2009, Proc. Fla. State Hort. Soc. 122:152-158）——数字来自下游文献转述，原文摘要未直接给出，引用前必须核原文 PDF。
5. **Qiu et al. 2022**（Frontiers in Plant Science，柑橘 HLB 症状级数据集）完整出版信息、DOI 与数据可得性未核实。这是农业侧最贴近的先例（"疑似/类似"分组本质是证据等级轴，且含煤污、溃疡），若可获取可作预训练数据源或标注规程模板。**建议单独追一次。**
6. **Sağlam et al. 2026**（J. Biomedical Informatics 181:105084）的 CA-AURC / CA-AUGRC 确切公式（ScienceDirect 403）。该文是"类别不平衡下的 risk-coverage 评测"最贴切的工作，对"黄龙病代价≫其他类"直接对口。
7. Xiang et al. 2026（PMC13236948）是否公开代码未确认（PMC 全文只说 PyTorch 2.x），若无代码需自行实现其基线。
8. 数字提取的置信度：partial-label 三篇对同一 baseline 报数存在正常的复现/设置差异（AN 在 COCO 上报 64.1/64.1/64.92；全标签上界报 75.8/78.0/76.78），**不可跨论文直接比较数字**，写入方案前须回原文核对。

**数据侧待核验**

9. **SciDB 两个 2026 数据集的实质内容**（最高优先级）：HDID 只知 21.59GB＋CC BY-NC 4.0；CTSD-YOLO-Split 只知 3.44GB＋YOLO 格式。SciDB 文件清单 API 全 404，Croissant 元数据 `contentUrl`/`contentSize` 为空，需人工浏览器登录 scidb.cn 查看。**HDID 的 NC 许可若项目有商业化预期是硬阻断，需先确认。**
10. **CitrusUAT 掩膜的实际标注对象**：文献称 953 张每张都有 ROI 二值掩膜，但标注的是"病斑区域"还是"整片叶片"**直接决定能否用于 IN-9 式前景/背景分离**（实验 A 需要整叶前景掩膜）。这是实验 A 能否零标注成本启动的前提，必须先下载核实。
11. BDLemonLeaf 的框标注格式（YOLO/COCO/XML 未标明）及"3431 张里有多少张真有框"未确认，需下载 96.2MB 抽样验证。
12. 孟加拉柠檬集（8d9fv6kpt3）文件端点未实测通过（落地页 200，files API 空、meta 504）；CitrusLeafVision（4dmgghmj3n）是否与其同源未确认；Zenodo 11148332 的 `orange_detection-COCO.rar` 是否即项目已有 Zenodo 检测集需比对。
13. Roboflow Universe 与 Figshare API 全站 403，本轮无法核验其柑橘条目。
14. MDPI 全站 403，MS-HLBD（声称 9219 张 5 类田间）未能核验。
15. ACP 木虱数据集 `access=restricted`，需向 ERATOSTHENES Centre of Excellence 申请。

**需本项目实测、无法从文献外推**

16. **点标注在 SciDB 上的可行性验证**（成本最低、说服力最强）：把已有框退化成中心点，跑 P2BNet / Group R-CNN，与已有 YOLOv8n mAP50=92.36% 直接对比。
17. **"病斑中心点"是否良定义**：COCO 物体有明确中心，病斑边界模糊且常成片融合（尤其斑驳黄化）。需在 190 张已有图上做点标注一致性实验（两位拍摄者各点一遍，测点间距分布）。若一致性差，P2BNet 的 proposal bag 构造会失效，届时需追 CPR++（arXiv 2401.17203，处理粗糙点标注）。
18. **点标注在手机 App 上的真实耗时**：文献的 0.9s／22.1s 均为桌面鼠标＋训练有素的 AMT 标注员。农民、触屏、田间光照下可能显著更高。需用两位拍摄者做小规模计时实验，此数字将成为论文成本论证的自有一手数据。
19. **CAM 在本项目数据上的真实 IoU**：为使排除论证从"引用他人"变为"自有证据"，建议在已有专家框的小子集上实测一次 Grad-CAM 的 MaxBoxAccV2（分类模型已有，成本极低）。
20. **专家—专家一致性上限**：所有 IoU/kappa 指标缺分母参照。若专家间病斑边界 IoU 只有 0.7，则 0.434 的弱监督 IoU 相对含义完全不同。
21. **1000 张规模下 partial-label 方法是否仍成立**：SPLC 的 τ=0.6、LL-R 的 Δ_rel=0.2 均在 VOC/COCO（1万–15万）上调出。小数据上模型很快记住全部样本，"大损失=假负例"的记忆效应假设可能失效。**必须先画已知正例与未观测项的损失分布直方图，确认两峰可分**；若不可分，应退回 Hill loss（不依赖记忆效应）。
22. **四轴联合训练是否真优于四个独立模型**：MTPSL 在 Cityscapes 上掩码 MTL **低于**单任务（69.50 vs 70.26）是明确反例，必须做对照。
23. **症状轴 vs 病名轴的证据权重是否应不同**：推断农民对"可见症状"的观察可靠性远高于其"病名诊断"可靠性，故建议只对病名轴施加强证据加权。**此为推断，未找到文献直接支持**，需按 kappa 矩阵方法实测，此实验决定权重设计，应优先做。
24. **黄龙病 PCR 样本的实际可获得数量**：方案中黄龙病节点只在 PCR 确证样本上算损失。若只有 10–30 例，该节点实际无法训练，只能作检索/相似度匹配而非分类输出，届时应明确改为"只输出斑驳＋黄化症状 ＋ 建议送检信号"。
25. **GLC 转移矩阵的条件数**：若证据等级×病名组合过多（如 4×15=60 维）而 trusted set 仅 100–200 例，T 的估计极不稳定；需实测条件数，过大则退回纯样本重加权。
26. **本项目自采数据是否存在采集偏差**（风险实在）：PlantVillage 一文证明采集偏差比语义背景偏差更隐蔽且去背景无法消除。**若公司 App 采集是"按病害类型分批外出拍摄"，则目标域数据本身带同类偏差，所有背景鲁棒方法的评测都会被污染。** 建议把 8 像素探针与模糊度探针同时施加于自采数据，并尽早向采集方确认拍摄批次与病害类别的对应关系。
27. **BG-Gap 在 5 类小数据上的统计显著性**：文献的 14.3% 来自 ImageNet 规模。1000 张/5 类下置信区间可能宽到无法区分方法差异，需 bootstrap 估计 CI。
28. **千张量级下"训练时方法 vs post-hoc"的排序**：OpenOOD v1.5 的规律是小数据集（CIFAR 5万）训练时方法胜约 2–3%、大规模反转，但**从未在 1000 张量级测过**。
29. **本项目 near-OOD AUROC 的实际落点**：文献参考区间 75–85，但柑橘缺素 vs HLB 的视觉相似度可能比 CUB 鸟种更极端（缺锌 vs 缺锌+HLB 阳性仅 92.86% 暗示可能更低）。
30. **分组划分 vs 随机划分的差值**：遥感给 28%，甘蔗叶给约 1.3 个 F1 点，量级差异大（取决于连拍冗余程度），须自行测量并作为结果之一报告。
31. **缺素样本的实际污染率**（混入 HLB 感染株的比例）：NINCO 的教训是可达 50%+，无 PCR 时只能报上界。
32. **专家（田间植保人员）在"缺素 vs HLB"上的实际错误率**：这决定 L2D 路线是否成立——若专家近 oracle，退化为纯选择性预测用阈值即可；若专家也常错，则应按 OvA-L2D 做，并可用"多标注员是否一致"作廉价标签。
33. **INT8 的 AP_small 分解必须自测**，不得引用他人聚合数字作为量化安全性依据。
34. **4060 Ti 实际 batch-VRAM 曲线**：RT-DETR-R18 的 11GB 是 3090 社区数字；DETR 系有训练中后期突然 OOM 的已知报告（有 4×16GB P100 跑到 epoch 13 才炸），启动时显存不代表全程。建议用 AutoBatch ＋ `torch.cuda.max_memory_allocated()` 自测。

## 对下游的影响

**方法方案 Agent 必须采用的**

1. **"分类＋弱定位 vs 检测"这个二选一应改为三选一**，且证据显示中间路线（点标注）成本接近左边（+20%）、性能接近右边（AP50 达 84%）。若仍要保留弱定位，只能作为可解释性审计工具，不能作定位输出。
2. **CAM 弱定位与自研 CAM 变体应从创新候选中移除**，理由是领域内实测 IoU 0.107 ＋ CVPR 2024 证明骨干可抹平所有变体收益。
3. **IRM／因果不变表征／复杂 DG 堆叠／朴素 Group DRO 应从方法池中移除**，替换为 ERM＋TrivialAugmentWide＋DFR 末层重训＋先源后目标微调。这也修正了 `PROJECT_STATE.md` 中"可选方法池"的域泛化条目。
4. **四轴标签体系已有完整可训练配方**，且黄龙病红线可编译为一个掩码条件。方案中应写明每轴的损失形式，而非只写"多轴标签学习"。
5. **可靠性评测的指标集须整体替换**：主指标 AUGRC／OSCR／SAC／Brier／classwise-ECE，禁用 ECE 作主指标、禁用 AURC/E-AURC 作主排序。
6. **"农业缺可靠性评测规范"不能再作为空白主张**，须改为引用 Frontiers 2026 作基座，空白收窄为"HLB 在 PCR 不可得下的可靠性评测"与"缺素作 near-OOD"。

**数据与标签 Agent 必须采用的**

7. HOCD-4 从数据规划中永久移除，不预留位置。
8. 柚类病虫害图像是全球性空白（仅一个条目且与已有数据同源）——**这是自采数据必要性的正面证据，可写入贡献陈述**。
9. 新增三个实测可下载的虫害源域候选（BDLemonLeaf v6、孟加拉柠檬集、CitrusFic），需补入 `DATA_MANIFEST.csv` 前先完成第 11–12 项核验。
10. **CitrusUAT 掩膜标注对象必须先核实**，它决定 IN-9 式诊断实验能否零标注成本启动。
11. 标注规程应参考 UF/IFAS EDIS PP328 的 "pen test"（HLB 斑驳黄化跨主脉不对称 vs 缺素两侧对称），并据此设"不可判定"类。
12. **必须尽早向公司确认 App 采集的拍摄批次与病害类别对应关系**（采集偏差风险）。
13. 数据划分必须按地块 > 树 > 连拍组 > 拍摄者/日期分组，且**同时报随机划分与分组划分两套数字**，差值即泄漏量；配 pHash 去重审计。
14. OOD 划分须按**胁迫类型**而非图片随机划分，且须设 full-spectrum 组（阴天/逆光/雨后仍算 ID）。

**实验 Agent 需要注意的（仍处 blocked，此处仅作预备）**

15. 最高优先级的低成本验证是"SciDB 框退化为点 → 跑 Group R-CNN → 与 YOLOv8n mAP50=92.36% 对比"。
16. 部署路线采纳"混合"方案：训练阶段照常用重模型做上界与蒸馏教师，不提前锁死；端侧转换走 PyTorch→PNNX→ncnn，优先 FP16／动态量化，静态 INT8 作最后手段且必须自测 AP_small。

## 下一步建议

1. **协调者先裁定第六节的判断修正**：确认"农业缺可靠性评测规范"这一空白主张作废，并把可辩护空白收窄为 HLB-无 PCR 可靠性评测与缺素作 near-OOD 两项。此项影响论文立意，应最先定。
2. **立即执行两个零成本／极低成本的自有证据实验**（不需要新数据、不需要专家）：
   - 8 像素探针 ＋ 模糊度探针，同时施加于 Mendeley、CitrusUAT 与 190 张自采图；
   - 在已有分类模型上实测 Grad-CAM 的 MaxBoxAccV2。
   这两项把"背景泄漏"和"CAM 不可用"从引用他人变为自有证据，一天内可完成，且直接构成论文动机章节。
3. **核实 CitrusUAT 掩膜标注对象**，若为整叶前景掩膜则立即启动 IN-9 八变体诊断实验（实验 A）。
4. **追一次 Qiu et al. 2022 的 DOI 与数据可得性**（第 5 项），其"疑似/类似 HLB"分组与本项目证据等级轴同构，若数据可得价值很高。
5. **发邮件申请 YOLO-Citrus 数据集**（华南农大，采集方式与本项目 App 高度一致）。
6. **人工登录 scidb.cn 核实 HDID 与 CTSD-YOLO-Split 的文件清单**，并先确认 HDID 的 CC BY-NC 许可是否与公司商业化预期冲突。
7. **向公司确认 App 采集协议中的批次—类别对应关系**，这是采集偏差风险的唯一前置控制点。
8. 待上述 2–3 项产出自有证据后，由协调者解锁方法方案 Agent，按"三选一（分类＋弱定位／点标注弱半监督／器官框检测）"重写候选方案排序。
