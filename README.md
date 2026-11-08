# Can a Quantized JEPA World-Model Planner Keep Its Task Success on a Neural Engine?

Code, pre-registrations, analysis outputs and per-episode outcomes for the paper "Can a Quantized JEPA World-Model
Planner Keep Its Task Success on a Neural Engine? A Paired Study of LeWorldModel" (Thomas Blanc, Independent
Researcher; CoRL 2026 Workshop on Efficient Foundation Models for Real-Time Embodied AI, non-archival; an extended
version adds the ternary experiment E21).

The study runs the predictor of LeWorldModel (LeWM) on the Apple Neural Engine through Core ML, compresses its weights
(16-bit, int8, 4-bit formats, GPTQ codes, mixed precision) and measures planning success on the simulated PushT and
TwoRoom tasks against PyTorch float32, with paired episodes, several planner seeds and pre-registered tests.

## Layout

Experiments are numbered E3 to E21 in the order they were run; in the files they are called EXP-003 to EXP-021.

- `main` holds the first experiment (EXP-003), this README, `data/`, `scripts/` and `COMMIT-MAP.txt`.
- Each branch `expNNN` holds the code frozen for experiment NNN: `run_grid.py`, the launcher `lancer_expNNN.sh`,
  `quant.py`, `gptq.py`, `coreml_pred.py` and other modules, the analysis script `analyse_expNNN.py` with its tests, and
  `PREREG-EXPNNN.md` with its dated amendments. Exceptions: `exp019` (`PILOTE-EXP019.md`, `lancer_pilote_exp019.sh`) and
  `exp020` (`CONSIGNE-AGENT.md`, `banc.py`) have no pre-registration and no analysis script.
- Code comments and pre-registrations are in French.

Where the governing pre-registration lives, when a later branch carries an older copy:

| Experiment | Branch to read |
| --- | --- |
| E3 | `main` (`PREREG.md`) |
| E10 | `exp010` (`PREREG-EXP010.md`); its statistics amendment is `AMENDEMENT-EXP010.md` in `exp012` to `exp018` |
| E11 | `exp011` (later branches keep an older copy with a 3-point margin) |
| E16 | `echecs`, French for failures (`PREREG-EXP016.md`, `analyse_exp016.py`) |
| E17 | `exp017-amend` (the `exp017` branch predates amendment 1) |
| E19, E20, E21 | `exp019`, `exp020`, `exp021`; `exp021-dev` holds the figure script written after E21's runs |

E19 is an exploratory pilot of E21 and E20 a study on a toy model; neither enters the paper's tests. Branches ending in
`-dev` or `-amend` were used to change files without stopping a running launcher.

## Data

`data/outcomes/<folder>.csv.gz` holds one row per run (episode x planner seed) for each results folder of the paper's
experiments; the file takes the folder's name, for example `exp018-gptq-pusht`. Columns:

| Column | Meaning |
| --- | --- |
| `file` | result file name (`<version>_s<seed>.npz`) |
| `spec` | version, as parsed by `quant.py` (`fp32` for full precision) |
| `seed` | planner seed |
| `position` | index of the episode in the experiment's draw |
| `episode_idx`, `start_step` | episode and start step in LeWM's dataset (start state and goal) |
| `success` | 1 if the simulator state reaches the goal |
| `first_success` | step of the first success, -1 if none |
| `engine` | `pytorch` (simulated quantization) or the Core ML engine (`coreml-ne`, `coreml-gpu`, ...) |
| `weights` | weight format on the chip (`int8`, `int4c`, `int4c-sp`, ...), or the same string as `spec` for PyTorch runs |
| `commit` | commit of this repository that produced the run (first 12 characters) |

The same `position` and seed share the same start state and random draws across versions, so outcomes are paired. A
folder can hold several runs of the same version with different seeds (for example a second float32 run, seeds 47 to
51). For Core ML runs, `spec` describes the PyTorch side only (`fp32`); read the format in `weights`.

`data/meta/<folder>.json`: each folder's run metadata (episode draw, planner configuration, engine and weight format).

`data/analyses/` holds the outputs of the frozen analysis scripts (`resultats.json`, `rapport.md`), the paper-wide Holm
sensitivity (`holm-article`), E13's isolated-call sweep, operation placement and energy validation
(`exp013-balayage-lot.json`, `exp013-placement-codages.json`, `exp013-validation-energie.json`) and the matrix behind the
figures (`vue-ensemble.json`). The paper's numbers are computed by a script from these files and the outcome tables.

The full result files of these runs (`.npz`, with states, poses and contacts, 0.43 GB) are attached to the repository's
release as `lewm-neural-engine-results-v1.tar.gz` (SHA-256 in the release notes). Extracted next to a checkout, they
form the `../resultats/` folder that the run and analysis scripts expect. Calibration and tuning data are not included.

## Checking the numbers

`python3 scripts/check_numbers.py` (standard library only) recomputes from `data/outcomes` the success rate of every
version and the paired mean differences behind E18's and E21's tests, and compares them with the frozen analyses'
outputs. It was written after the analyses, for readers.

The frozen analysis scripts also check that the code commit recorded in each result file exists and is unchanged. This
public copy rewrote author addresses and co-author lines, which changes every commit hash, so those checks fail here
although file contents are identical. `COMMIT-MAP.txt` maps each recorded hash to this repository's commit. The E21
analysis test suite (`test_analyse_exp021.py` on branch `exp021`) runs unchanged once the release archive is extracted.

## Reproducing

The runs used LeWM's released checkpoints through stable-worldmodel, PyTorch, and coremltools 9.0 on a MacBook Pro M5
Max under macOS 26.6.2 (E19 to E21: macOS 27.2, simulation only). Each launcher (`lancer_expNNN.sh`, or `lancer_grille.sh` for E3) gives the exact commands,
including a short trial mode, the controls that stop a run (reproduction of an earlier reference, protocol checks), and
the analysis call. Paths to the local LeWM installation are set at the top of `run_grid.py` and the launchers.

## License

The code is under the MIT license (`LICENSE`) and the data in `data/` under CC BY 4.0 (`data/LICENSE.md`).

## Notes

- In this public copy, author and committer addresses use a GitHub no-reply address and commit messages carry no
  co-author lines; dates, titles and file contents are unchanged. The dates were recorded on the author's machine and are
  not externally timestamped.
- AI assistants helped write the code, the text and the audits; the author designed the study, drew the conclusions and
  checked all AI output.
