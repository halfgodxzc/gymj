# GYMJ: Guiyang Mahjong AI

[中文](README.md) | English

GYMJ is a four-player Guiyang Mahjong rules engine with an R5 policy model and self-play training tools. The engine determines legal moves, advances play, and settles each player's score. R5 chooses among legal moves using only its own hand and public information. Playing a hand requires neither a critic nor access to opponents' concealed tiles or the wall.

## Rules and training

- [Rules and scoring](docs/guiyang_rules.en.md) ([中文](docs/guiyang_rules.md)): when you can win on a discard, how chickens and kongs score, and who pays when the wall runs out.
- [How R5 was trained (Chinese)](docs/r5_basic_training.md): rule-based demonstrations, examples of higher-scoring hands, critic pretraining, PPO self-play, and evaluation against fixed opponents.

## Repository layout

| Location | Contents |
| --- | --- |
| [libgy/src](libgy/src/) | Rules, round settlement, feature encoding, and Python bindings |
| [python/gymj](python/gymj/) | Agents, data collection, actor and critic models, training, and evaluation |
| [examples](examples/) | Start with `play_r5.py` to play one hand |
| [docs](docs/) | Player rules and training guides |
| [tests](tests/) · [benchmarks](benchmarks/) | Correctness tests and performance benchmarks |
| [experiments](experiments/) | Historical training configurations and model records; weights and sample games are in [Releases](https://github.com/halfgodxzc/gymj/releases) |

## Build from source

The tested configuration is Windows x64 with Visual Studio 2022 and Python 3.13. You need a C++17 compiler, CMake, and Ninja. From the repository root, run the following in a Visual Studio Developer PowerShell with your Python virtual environment active:

```powershell
python -m pip install -r requirements.txt
$py = python -c "import sys; print(sys.executable)"
$bind = python -m pybind11 --cmakedir
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE=/O1 -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DGYMJ_BUILD_PYTHON=ON "-DPython_EXECUTABLE=$py" "-Dpybind11_DIR=$bind"
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
$env:PYTHONPATH = Join-Path $PWD python
```

## Play one hand on CPU

Place the matching `actor.pt` in the repository root. Compare its SHA256 digest with the entry for `actor.pt` in the release bundle's `SHA256SUMS.txt` before loading it:

```powershell
Get-FileHash actor.pt -Algorithm SHA256
python examples/play_r5.py actor.pt --seed 42
```

All four seats use the same frozen policy. The example prints the wall seed, number of decisions, how the hand ended, and the four seats' net scores. These are actual settlement scores, not model predictions: they sum to zero.

## What is in the release candidate?

The candidate includes the R5 actor and 24 machine self-play hands for testing data loading and the training pipeline. It does not include a paired critic, the web service, or human play logs. The sample is not large enough to reproduce R5's playing strength.

This checkpoint is a later R5 variant, not the original fifth-round PPO checkpoint. It includes a kong adapter and was fine-tuned on discard choices using a 50/50 blend of the previous policy's probabilities and rule-based targets that favor more effective draws. It was trained before the passed-pung restriction was added. The updated engine now excludes those illegal calls; completing a hand confirms compatibility, not improved playing strength under the new rule.

## Licenses

| Material | License |
| --- | --- |
| Original project code | [Apache-2.0](LICENSE). Commercial use is permitted. |
| R5 weights (`actor.pt`) | [CC BY-NC 4.0](LICENSES/CC-BY-NC-4.0.txt). Noncommercial use, sharing, and adaptation with attribution; commercial use requires separate permission. |
| Player rules document and sample self-play data | [CC BY 4.0](LICENSES/CC-BY-4.0.txt). Commercial use is permitted with attribution. |

For the Creative Commons materials, credit the GYMJ project, link to the license, and indicate any changes. Third-party dependencies and assets retain their own licenses.
