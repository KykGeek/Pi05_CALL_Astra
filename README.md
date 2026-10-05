# CALL LLM + Pi0.5 for LIBERO

This repository contains the CALL LLM evaluation runner and recovery integration for an existing Pi0.5-controlled LIBERO episode. Astra is the configured model; `call_llm/` names the host-side language-model control and API integration. This is research code, not a standalone simulator or a packaged policy checkpoint.

## Repository map

| Path | Purpose |
| --- | --- |
| `call_llm/` | Pi0.5 CALL decision logic, feature helpers, and language-model-assisted recovery controller. |
| `call_llm/runtime/` | Responses client, tool protocol, EEF adapter, execution journal, and step ownership. |
| `scripts/benchmark/run_libero_call_llm_benchmark.py` | Main episode/task benchmark runner. |
| `scripts/benchmark/run_shadow_suite_pipeline.sh` | Optional multi-suite shadow-evaluation pipeline. |
| `scripts/diagnostics/` | Provider and tool-schema probes plus the LIBERO action-protocol audit. |
| `docs/prompts/call_llm_recovery_instructions.md` | English developer instructions supplied to the model. |
| `history/` | Preserved duplicate/obsolete files for reference; not imported by the runtime. |

## Runtime requirements

The benchmark runner expects an existing OpenPI checkout, a compatible Pi0.5 LIBERO checkpoint, and a LIBERO/robosuite environment. These external assets are deliberately not bundled here. Configure their paths with `OPENPI_ROOT` and `PI05_CHECKPOINT`, or pass the corresponding command-line options. Astra provider credentials must be supplied through the supported Codex/provider environment; never commit credentials.

The benchmark needs a Linux environment with the project’s robotics dependencies and an available GPU. The scripts are not expected to run on a plain Windows Python installation.

## Run

From the repository root on the configured evaluation host:

```bash
export OPENPI_ROOT=/path/to/openpi
export PI05_CHECKPOINT=/path/to/pi05_libero_checkpoint
python scripts/benchmark/run_libero_call_llm_benchmark.py --help
```

Use the runner’s `--suite`, `--episodes-per-task`, `--output-dir`, and GPU options to select a benchmark. Review `--help` on the target host before launching an evaluation; this integration can control a simulator and may call an external model API.

To run the optional four-suite pipeline, configure `MODEL_ROOT` to point to trained CALL LLM decision heads, then run:

```bash
bash scripts/benchmark/run_shadow_suite_pipeline.sh
```

## Diagnostics

The scripts under `scripts/diagnostics/` probe provider connectivity/tool schemas or audit the LIBERO action mapping. They do not replace a full benchmark run. The robot action audit requires the same external LIBERO/OpenPI environment as the runner.

## Project status

This is an extracted experiment integration. Some training artifacts, external datasets, and environment setup are maintained outside this repository. No open-source license has been selected yet; add a `LICENSE` file before distributing the code as an open-source project.
