# Training GYMJ: From a Rule-Based Teacher to the Public R5

[中文版](r5_training_retrospective.md) | November 2026 | [Training parameters and scripts (Chinese)](r5_basic_training.md)

In Guiyang Mahjong, the player who completes a winning hand can still lose points. Whether a player finishes ready to win, who first discards a scoring tile known as a *chicken*, who claims it, and who deals into a win after a kong can all change the settlement. Our objective was therefore not to win each hand as quickly as possible, but to improve average net points using only the information available to the player. The engine tells us exactly how much each seat earned. It does not tell us how much of that result came from a discard ten turns earlier.

## 1. Learning the basics without human game records

Our first agent played randomly. Its purpose was to test the full loop: could the engine offer legal actions, play a hand to completion, and assign each action and reward to the right seat? Random play rarely produced a winning hand, so we wrote a teacher. It tried each legal discard and favored the lowest *shanten* (distance to a ready hand). When offered a pung, it compared passing with claiming the tile and making the best subsequent discard.

V2 learned by imitating this teacher. With V3 we tried self-play PPO, comparing a shanten-based auxiliary reward with net points alone and varying the learning rate and exploration settings. The teacher provided a reliable way to approach a ready hand, but speed and profit are not the same objective. A larger update can move action probabilities further without discovering a strategy that the model almost never tries. We also changed how we compared policies: use the same walls, rotate seats, and report per-seat net points and confidence intervals separately for each opponent.

## 2. V4: Representing the alternatives behind a discard

Two discards can both leave a hand two steps from ready, yet one leaves more useful draws or preserves a route to seven pairs. We added candidate-action features for standard hands, seven pairs and all triplets; counts of distinct improving draws and their remaining copies; and distance to a single-suit hand. Adding these as a residual branch on V3 changed few decisions. Amplifying the branch did not yield a consistent gain either. Supplying a feature and showing that the policy actually uses it are separate tests.

V4 redesigned how states and candidate actions were scored rather than simply continuing from V3 weights. After imitating the basic teacher, it also learned from BigHandAgent, a rule-based player that pursued seven pairs, all triplets or a single suit. One main supervised run mixed **3,500 basic-teacher hands with 1,500 big-hand demonstrations**. This exposed the policy to routes the conservative teacher rarely took while retaining examples of ordinary play. Imitation shows the learner what the demonstrator chose; it does not establish that the learner knows when a higher-scoring hand is worth the delay.

## 3. From V4 to the original R5

Before PPO, we pretrained a critic on **5,000 complete hands** to predict two quantities separately: the seat's final net points and its future auxiliary shanten return. During training the critic could see opponents' concealed hands and the wall. The actor that actually picked moves saw only its own hand and public information; it did not need the critic at deployment.

We then ran **five PPO rounds of 1,000 fresh hands each**, using the current actor to generate each round. The return combined the seat's final net points divided by 30 with 0.25 times its normalized future shanten return. An observed return above the old critic's expectation tended to increase the probability of the sampled action; a return below expectation tended to reduce it. We evaluated each round before selecting the next starting point. The fifth-round actor was the **original R5**. These were not pure net-score runs, and the 5,000 PPO hands exclude both supervised data and critic pretraining.

Under the rules and fixed evaluation protocol used at the time:

| Matchup | Hands | Mean net points per R5 seat |
| --- | ---: | ---: |
| Two R5 seats vs. two frozen V4 seats | 1,000 | +2.54 |
| Two R5 seats vs. two rule-based teachers | 1,000 | +2.40 |

The interval for the incremental gain from round four to round five included zero. The table is not evidence that every update improved the policy. A separate count exposed a weakness neither average conveys: **R5 chose a kong zero times in 1,328 legal opportunities.**

## 4. Looking beyond the average score

**Kongs.** Unlike a pung, a kong introduces an unknown replacement tile. We enumerated possible replacements from the player's perspective and estimated how often shanten would improve, stay unchanged or worsen. A collection policy tried both taking and declining kongs, giving us examples for a kong adapter and subsequent joint PPO experiments. This addressed a gap in action coverage; it did not make the heuristic an oracle for when to declare a kong.

**Effective draws.** We selected decisions where two discards left the same shanten but different numbers of improving draws. From the same decision point we tried the alternative and played the hand to completion. This made the comparison more informative than contrasting the scores of unrelated hands. We then used soft targets to adjust discard preferences at selected positions and retained the previous policy's targets elsewhere. The continuations tested the direction of the change; the supervised rule was not “copy whichever branch happened to win once.”

**Chickens and pungs.** Human games helped locate disagreements, not supply infallible labels. For example, about 94% of the first fixed-chicken discards in one analysis ultimately counted as charging chickens. That answers “what happened *after* R5 discarded one?” It does not answer “how often did R5 discard one *when it could*?” The second question needs all eligible opportunities as its denominator, including those where R5 chose another tile. We compared choices at similar shanten, tried opponents more willing to pung, attributed chicken bonuses to specific actions, and ran paired continuations for claiming versus passing. These tests identified decisions that changed, but did not establish that a particular bonus consistently improved net points against humans.

**Value prediction.** A critic estimates the average return from a position, not the outcome of one particular continuation. We replayed fixed positions multiple times and compared predictions with their mean returns. We also tested separate settlement components, four-seat zero-sum predictions and action-conditioned value estimates. A prediction can miss a single outcome by a wide margin yet be close to the continuation mean. Conversely, enforcing zero sum does not by itself make the actor pick a better discard. Prediction quality and playing strength need different tests.

## 5. What the public checkpoint represents

The [v0.1.0 checkpoint](https://github.com/halfgodxzc/gymj/releases/tag/v0.1.0) is **R5 with a kong adapter and effective-draw soft-target fine-tuning**, not the original fifth-round actor in the evaluation table. At selected teaching positions, the target mixed the previous policy distribution and the rule favoring more improving draws in equal proportions. Samples used to preserve behavior elsewhere kept the previous policy distribution. The release does not combine every subsequent PPO, critic, search or replay experiment. We verified that it could complete a hand after the rule change restricting a pung after a player had passed on the same tile; that compatibility check was not a new strength benchmark.

The teacher taught basic play without expert logs. Self-play produced measurable net-score gains against specific opponents under a fixed protocol. Counting opportunities and replaying alternatives then revealed weaknesses the headline scores had missed. We do not yet have evidence that R5 consistently beats skilled human players or has mastered the trade-offs between charging a chicken, pursuing a high-scoring hand and staying ready to win.
