# R5 对局与改进实证 / Human matches and post-R5 experiments

[English](#english) · [训练沿革](r5_training_retrospective.md)

## 真人究竟赢了多少？

2026-10-31 的审计快照覆盖同一网页模型（进张软标签 50% 版，权重 SHA256 前缀 `9129dcc5`）。109 局有真人操作，103 局完成、6 局中止；输赢只计完局。共 103 副不同牌墙、41 个会话、11 个昵称，最近真人操作为 10 月 22 日。这不是全站实时累计数。每局为一席真人和三席各自计分的 R5，不是三席合作对抗真人。

| 指标 | 真人（103 席局） | R5（三席共 309 席局） |
|---|---:|---:|
| 累计净分 | +718 | −718 |
| 每席每局净分 | +6.97 | −2.32 |
| 胡牌 | 32/103（31.07%） | 65/309（21.04%） |
| 放炮 | 5/103（4.85%） | 16/309（5.18%） |
| 有鸡结算的终局有叫，含胡牌者 | 69/96（71.88%） | 183/288（63.54%） |

真人 **59 局正分、41 局负分、3 局零分**。正分率 57.28% 不是胡牌率，也不表示那 59 局三席 R5 各自都输钱。按会话聚类，真人平均净分的描述性 95% 区间为 **[+3.66, +10.67]**。剔除场次最多的一位，其余 63 局仍为 +414（每局 +6.57）。但每套首局真人固定坐庄，真人能看到 AI 推荐，参与和退出并非随机。这批真人占优，不能外推为所有玩家对 R5 的胜率。

### 输在哪些分项？

| 账本项目 | 真人净收益 | R5 三席合计 |
|---|---:|---:|
| 胡分 | +155 | −155 |
| 鸡分 | +523 | −523 |
| 责任鸡 | +40 | −40 |
| 三种豆分 | −15 | +15 |
| 荒庄查叫 | +15 | −15 |
| **合计** | **+718** | **−718** |

鸡与责任鸡 **+563，占真人净优势的 78.4%**。其中 +523 鸡分拆为冲幺鸡 +170、冲八筒 +135、普通幺鸡 +26、普通八筒 +152、其他翻牌鸡 +40。金鸡倍率已经包含在分项里。这里统计的是收付相抵的结果，不是给某个弃牌动作估出来的收益。

### 同一个局面，人和模型哪里不同？

- **冲鸡：** 冲与不冲都能保持最低向听时，真人冲了 47/197 次（23.9%），同局面 R5 推荐冲 46/197 次（23.4%）。频率很接近，不表示每次选择都一样，也不支持仅凭账本断言 R5 是因为“不够爱冲”而输分。
- **碰牌：** 碰后向听不变时，真人碰 30/59 次（50.8%），R5 只推荐碰 8/59 次（13.6%）。这是比冲鸡频率更明显的行为分歧，但人的动作本身不是最优标签。
- **有叫：** 较早的 101 完局深审是上述样本的子集，不可相加。即使排除胡牌者，在有鸡结算的牌局中，真人仍有叫 37/63（58.7%），R5 为 115/218（52.8%）。因此差异不只是“胡得多的人必然算有叫”。
- **进张：** 在该子集中，195 次不同弃牌的普通型、七对、大对子向听均匹配；R5 推荐保留更多有效进张 74 次，真人更多 48 次，两者相等 73 次。不能把早期模型忽略进张的诊断套在这个网页版本上。

## R5 之后：哪些改动有实测收益？

每行只与本实验自己的对照比较。净分为每席每局；“配对增量”是相同牌墙的实验组减对照组，不是绝对净分。方括号为 95% 区间。起点、规则、对手或采样方式并非都相同，不能把下列结果相加。

| 改动 | 比较与收益 | 可以得出的结论 |
|---|---|---|
| 杠适配器及联合 PPO | 原始 R5 有 1,328 次合法杠机会却从未选杠。两个后续适配版直接交手，各方两席在 1,000 局分别杠 320、317 次；长程候选对上一版 +0.356 [−0.187, +0.909]。 | 不再完全回避杠。两个适配版直接对战不能单独估计适配器赚了几分。 |
| 同向听进张软标签 | 对微调前同一 R5：10% 版 +0.101 [−0.439, +0.636]；**50% 版 +0.998 [+0.295, +1.702]**。50% 直接对 10% 为 +0.983 [+0.238, +1.730]。每组 1,000 局、167 副独立牌墙轮座。 | 确认了固定对手收益。50% 对旧版的鸡及责任鸡分项是 +0.890。真人主表对应这个 50% 版，不是微调前的版本。 |
| 关闭向听奖励，只按净分 PPO | 网页版起步 2,000 局后对其起点 +0.270 [−0.069, +0.638]；另一分支 6,000 局自博弈后对自身起点 −0.162 [−0.722, +0.381]。 | 都未确认超越起点，也不是只改变数据量的消融。6,000 局分支同向听碰鸡平均概率从 9.51% 升至 11.99%，贪心碰次数却仅从 114 增至 116。 |
| 有条件提高碰概率，不改权重 | 只在仅可碰/过、碰后能维持向听且存在增加进张的弃牌时干预。每组 1,000 副相同牌墙，一席干预对三席冻结 R5，温度 1。碰概率至少 50% 较原策略 **+1.636 [+0.934, +2.340]**；符合条件必碰为 **+1.873 [+0.922, +2.822]**。 | 终局有叫率分别增加 2.20、2.90 个百分点。必碰与 50% 之间未分出高下。这是整局脚本干预收益，不是已训练进网页权重的收益。 |
| 给鸡相关动作额外奖励 | 4,000 局混合采样后，0.5 组对冻结 R5 的训练前后配对变化 −0.412 [−1.093, +0.246]，对 Teacher **−1.160 [−2.040, −0.232]**；1.0 组对冻结 R5 为 −0.089 [−0.641, +0.455]。 | 未得到稳定净收益。各组起点与样本量不同，不能断言某个权重在所有条件下有害。 |
| 碰杠成对续打、校准 Critic 后联合 PPO | 辅助数据每碰续打 16 次、每杠 8 次；另采 2,000 局校准和训练。自身决策净分 MSE 从 0.42746 降至 0.42378（净分/30 的平方单位）。对直接起点 800 局 +0.094 [−0.607, +0.799]；对 Teacher 200 局 +6.440 [+4.822, +8.055]。 | 误差改善约 0.86%，配对区间也跨零。赢 Teacher 不等于比训练前强；没有同批不校准对照，不能把变化单独归给 Critic。 |

**结论：** 进张微调得到的**模型权重**赢过直接起点；有条件多碰的**规则干预**在随机采样对局中也有收益。两者不是同一个成果。真人记录定位到了鸡分、有叫和同向听碰牌的差距，但没有同口径的人机前后测证明后续改动已经缩小这个差距。

<a id="english"></a>

## English: Human matches and post-R5 experiments

The October 31 audit covers one web checkpoint (`9129dcc5`, the 50% effective-draw model). Of 109 hands with human input, 103 finished and six were abandoned. Each hand had one human and three independently scoring R5 seats. Humans finished positive 59 times, negative 41 times and flat three times: a **57.28% positive-score rate**, distinct from a **31.07% hand-win rate**. Humans earned **+718 total, +6.97 per seat-hand** (session-clustered 95% interval [+3.66, +10.67]); R5 earned −718 total, −2.32 per seat-hand. Humans won 32/103 hands versus 65/309 R5 seat-hands; deal-ins were 5/103 versus 16/309. The 41 sessions and 11 screen names were not a random player sample. Humans started each set as dealer and could see AI recommendations. This was not a blind, seat-balanced benchmark.

Human net payments were +155 for winning hands, +523 for chickens, +40 for responsibility chickens, −15 for kongs and +15 for exhaustive-draw settlement. Chicken-related payments accounted for **78.4% of the observed lead**, not 78.4% of decisions. At matched minimum-shanten opportunities, humans charged 47/197 times versus 46/197 R5 recommendations. For shanten-preserving pungs the difference was much larger: 30/59 versus 8/59. Humans finished ready more often in hands with chicken settlement (69/96 versus 183/288, including winners). These observations identify decisions worth testing; they do not establish that every additional pung pays off.

| Experiment | Measured result and scope |
|---|---|
| Kong adapter | Original R5 chose zero of 1,328 eligible kongs; two later adapter policies took 320 and 317 in a 1,000-hand matchup. The newer policy scored +0.356 [−0.187, +0.909] per seat. Behavior changed; the incremental strength gain was inconclusive. |
| Effective-draw soft labels | Against the same pre-tuning policy, 10% labels scored +0.101 [−0.439, +0.636], **50% labels +0.998 [+0.295, +1.702]**. Each matchup used 1,000 hands on 167 walls with seat rotation. The 50% checkpoint was the one in the human audit. |
| Score-only PPO | +0.270 [−0.069, +0.638] against the parent after 2,000 hands; a separate 6,000-hand branch scored −0.162 [−0.722, +0.381] against its own parent. Neither established an improvement. |
| Conditional pung intervention | When only pung/pass were legal and a post-pung discard preserved shanten while increasing improving copies, a 50% pung-probability floor gained +1.636 [+0.934, +2.340] versus unchanged play; always claiming gained +1.873 [+0.922, +2.822]. Paired 1,000-wall experiments, one intervention seat against three R5 seats at temperature 1; not a learned checkpoint. |
| Chicken-action bonuses | After 4,000 mixed-policy hands, the 0.5 branch changed by −0.412 against frozen R5 and −1.160 [−2.040, −0.232] against Teacher relative to its previous scores. The 1.0 branch changed by −0.089 against frozen R5. No consistent gain. |
| Paired call continuations and critic calibration | With 16 repeats for pungs and eight for kongs, followed by a 2,000-hand calibration/PPO run, own-seat decision MSE fell about 0.86%. The actor scored +0.094 [−0.607, +0.799] against its parent over 800 hands. Neither the critic nor the actor result established an improvement over the parent. |

No comparable human rematch has established that the later experiments closed the human score gap. Improvements across historical branches are not additive.

**Audit references (local run IDs):** `r5-existing-data-audit-20261031`, `web-human-r5-deep-20261018`, `kan-adapter-r5-joint-ppo-long-2000-20261010`, `r5-effective-draw-bc-alpha50-paired-20261013`, `r5-alpha50-score-only-ppo-2000-20261014`, `r5-selfplay-6000-multiepoch-1000-20261015`, `r5-pon-intervention-3x1000-20261031`, `chicken-mixed-4000-20261109`, `call-critic-calibration-ppo-2000-20261114`. Only aggregates are included; no player names or private game records.
