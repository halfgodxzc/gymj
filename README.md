# GYMJ：贵阳麻将引擎与 R5

贵阳麻将 AI：四人捉鸡麻将规则引擎、R5 策略模型与自博弈训练工具。提供可下载的模型权重、对局样例和中文玩法说明。

A four-player Guiyang Mahjong rules engine, R5 policy model, and self-play training toolkit, with pretrained weights and sample games.

中文 | [English](README.en.md)

GYMJ 提供四人贵阳麻将规则引擎和策略模型。Arena 负责合法动作、牌局推进和四家结算；R5 根据本家手牌和公开信息选择动作。普通推理不需要知道牌墙或对手暗手，也不需要 Critic。

- [玩法与计分](docs/guiyang_rules.md)：报听、点炮资格、过碰、冲锋鸡、豆分和荒庄查叫。
- [R5 的训练过程](docs/r5_basic_training.md)：教师示范、大牌混合监督、价值预训练、自博弈 PPO 与评估。

## 目录导航

| 位置 | 内容 |
| --- | --- |
| [libgy/src](libgy/src/) | 麻将规则、牌局结算、特征编码和 Python 接口 |
| [python/gymj](python/gymj/) | Agent、数据采集、Actor/Critic、训练和评估 |
| [examples](examples/) | 从 `play_r5.py` 开始，用权重打一局 |
| [docs](docs/) | [英文玩法规则](docs/guiyang_rules.en.md)、中文规则和训练说明 |
| [tests](tests/) · [benchmarks](benchmarks/) | 正确性测试与性能测试 |
| [experiments](experiments/) | 历史训练配置与模型版本记录；权重和样例在 [Releases](https://github.com/halfgodxzc/gymj/releases) |

[训练回顾：从规则教师到公开的 R5](docs/r5_training_retrospective.md)

[真人对局与后续实验：R5 输在哪里，哪些改动有收益](docs/r5_match_results.md)

## 从源码运行

已验证的平台是 Windows x64、Visual Studio 2022、Python 3.13。需要 C++17 编译器、CMake 与 Ninja。以下命令在已激活 Python 虚拟环境的 VS 开发者 PowerShell 中，从仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
$py = python -c "import sys; print(sys.executable)"
$bind = python -m pybind11 --cmakedir
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE=/O1 -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DGYMJ_BUILD_PYTHON=ON "-DPython_EXECUTABLE=$py" "-Dpybind11_DIR=$bind"
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
$env:PYTHONPATH = Join-Path $PWD python
```

## 用权重打一局

将配套的 Release 附件 `actor.pt` 放在当前目录，先用 `Get-FileHash actor.pt -Algorithm SHA256` 与清单核对，再运行：

```powershell
python examples/play_r5.py actor.pt --seed 42
```

四个座位共享同一个冻结 Actor。输出依次给出牌墙种子、决策次数、结束原因和四家净分。净分是各家实际结算，不是模型打分；四家之和应为零。

## 首发候选的范围

**许可按内容区分：** 本项目原创代码采用 [Apache-2.0](LICENSE)，允许商用。Release 中的 R5 权重 `actor.pt` 采用 [CC BY-NC 4.0](LICENSES/CC-BY-NC-4.0.txt)，可署名后非商业使用、分享和改编；商业使用须另行取得授权。[玩法规则文档](docs/guiyang_rules.md)及 Release 中的机器自博弈样例数据采用 [CC BY 4.0](LICENSES/CC-BY-4.0.txt)。署名可写“GYMJ 项目”，请附许可证链接并注明改动。第三方依赖和素材不因本说明而改变原有许可。

候选权重是带杠适配器的 R5 进张软标签微调版（规则标签占 50%），不是最初第五轮 PPO 的同名模型。它在过碰规则加入之前训练；现在由新规则屏蔽不合法的碰。能跑完整局不等于已证明它在新规则下更强。示例数据仅供读取、调试和训练冒烟，不足以复现历史牌力。首发候选不含网页服务、真人日志或所有探索实验。训练新的 Critic 可参考训练说明，不要把无关实验的 Critic 当作配套权重。
