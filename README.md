# CALL LLM + Pi0.5 for LIBERO

This repository contains an experimental CALL LLM evaluation runner and recovery integration for Pi0.5-controlled LIBERO episodes. Astra is the configured model; `call_llm/` names the host-side language-model control and API integration. This is research code, not a standalone simulator or a packaged policy checkpoint.

## Repository map

| Path | Purpose |
| --- | --- |
| `call_llm/` | CALL decision logic, feature helpers, and language-model-assisted recovery. |
| `call_llm/runtime/` | Responses client, tool protocol, EEF adapter, execution journal, and step ownership. |
| `scripts/benchmark/run_libero_call_llm_benchmark.py` | Main task/episode benchmark runner. |
| `scripts/benchmark/run_shadow_suite_pipeline.sh` | Optional multi-suite shadow-evaluation pipeline. |
| `scripts/diagnostics/` | Provider/tool-schema probes and LIBERO action-protocol audit. |
| `docs/prompts/call_llm_recovery_instructions.md` | Developer instructions supplied to the model. |
| `history/` | Archived reference material; not imported by the runtime. |
| `requirements.txt` | Python package dependencies used by the code in this repository. |

## Requirements

The full benchmark must run on a Linux host configured for the matching OpenPI, Pi0.5, LIBERO, robosuite, and GPU stack. This repository does not include the simulator, datasets, model checkpoint, or OpenPI source checkout. Install OpenPI and its robotics dependencies using their own instructions; their JAX/PyTorch and CUDA versions must be compatible with the selected GPU and checkpoint.

Install this repository's Python dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

The requirements file lists package names without version pins because the correct JAX, PyTorch, and CUDA builds depend on the external OpenPI environment. For reproducible experiments, record the exact environment versions used on the evaluation host.

You also need:

- An OpenPI checkout and a compatible Pi0.5 LIBERO checkpoint.
- The LIBERO benchmark assets and a working robosuite/MuJoCo rendering setup.
- The Codex app-server executable available to the runtime for model-assisted recovery.
- Provider credentials configured in the host environment. Never commit API keys or `.env` files.

## Configure and run

Set the external OpenPI and checkpoint paths, then inspect the runner options:

```bash
export OPENPI_ROOT=/path/to/openpi
export PI05_CHECKPOINT=/path/to/pi05_libero_checkpoint
python scripts/benchmark/run_libero_call_llm_benchmark.py --help
```

Use the runner's `--suite`, `--episodes-per-task`, `--output-dir`, and GPU options to choose an evaluation. Review `--help` on the target host before launching: the runner controls a simulator and may call an external model API.

For the optional four-suite shadow pipeline, configure `MODEL_ROOT` to point to trained CALL LLM decision heads, then run:

```bash
bash scripts/benchmark/run_shadow_suite_pipeline.sh
```

## Diagnostics

Scripts in `scripts/diagnostics/` probe provider connectivity/tool schemas or audit LIBERO action mapping. They do not replace a full benchmark run. The action-protocol audit requires the configured external LIBERO/OpenPI environment.

## Limitations and distribution

This is an extracted research integration. Training artifacts, external datasets, simulator assets, and environment setup are maintained separately. A project license has not yet been selected; add a `LICENSE` file before distributing this repository as open-source software. Also verify third-party asset and dependency licenses before redistribution.
