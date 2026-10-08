# π0.5 + CALL + Astra for LIBERO

[Project website, paper, and interactive episode demos](https://kykgeek.github.io/Pi05_CALL_Astra/)

The website is served from `docs/` by GitHub Pages. See [website deployment notes](WEBSITE.md) for maintenance and release verification.

Research runtime for a LIBERO episode controlled by π0.5, with a learned CALL head and an optional Astra recovery controller. The runtime includes the verified handoff/re-entry gate, the LIBERO EEF adapter, bounded Astra execution, and episode logging.

## Scope

This repository contains integration code, not model weights or a complete robotics distribution. OpenPI, LIBERO/robosuite, the π0.5 checkpoint, trained CALL heads, and provider credentials are external by design. Configure their locations and credentials on the evaluation host; do not add them to Git.

The live controller uses public images and robot observations for CALL/Astra inputs. Simulator-private object state is not used as a policy feature. Private simulator snapshots may be retained locally for audit; generated snapshots and experiment results are excluded from this source package.

## External runtime inputs

Prepare a compatible Linux/GPU environment that already has the OpenPI checkout and its LIBERO dependencies. This project deliberately does not install or pin OpenPI, JAX, CUDA, MuJoCo, or the π0.5 weights.

The runner accepts these external paths:

- `OPENPI_ROOT`: OpenPI checkout, including its `third_party/libero` tree.
- `PI05_CHECKPOINT`: compatible LIBERO π0.5 checkpoint directory; its directory name must be `pi05_libero`.
- `--model-dir`: trained CALL-head directory. For the learned baseline it must contain the expected `fold_models/fold_*.pt` checkpoints and checkpoint metadata.

The Astra transport is provider-configurable. Set these in the process environment or a secret manager, never in a tracked file:

- `ASTRA_CODEX_PROVIDER`: provider identifier used by Codex CLI.
- `ASTRA_BASE_URL`: provider Responses API base URL, ending in `/v1`.
- `ASTRA_API_KEY_ENV`: name of the environment variable holding the secret; defaults to `ASTRA_API_KEY`.
- `ASTRA_MODEL`: model identifier; defaults to `gpt-6-astra`.
- `ASTRA_REASONING_EFFORT`: reasoning effort; defaults to `medium`.

The `codex` CLI must be installed and available on `PATH`. The API key itself is read from the environment variable named by `ASTRA_API_KEY_ENV`; it is never written into the repository. Do not commit `.env` files. If a credential was ever committed, rotate it rather than relying on deleting the file.

## Install and verify

Run these commands inside the already-compatible OpenPI environment. They install only this integration's small runtime additions, not OpenPI or its GPU stack:

```bash
python -m pip install -r requirements-runtime.txt
python scripts/run_call_astra_v1_closed_loop.py --help
```

Verify the live runtime with a one-episode simulator smoke test on the target host after configuring the external assets and provider credentials.

## One-episode live smoke test

After configuring the external assets and provider credentials, select one task and a new output directory. The forced-call switch is diagnostic only: it deliberately requests one Astra intervention and should not be used for formal evaluation.

```bash
python scripts/run_call_astra_v1_closed_loop.py \
  --openpi-root "$OPENPI_ROOT" \
  --checkpoint "$PI05_CHECKPOINT" \
  --model-dir "$CALL_MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --suite libero_10 --task-ids 0 --episodes-per-task 1 \
  --baseline learned --threshold-point conservative \
  --decision-rule 4_of_5 --cooldown-queries 6 \
  --enable-astra --astra-smoke-force-call \
  --astra-model "$ASTRA_MODEL" \
  --astra-reasoning-effort "$ASTRA_REASONING_EFFORT" \
  --astra-max-decisions 25 --astra-max-execution-chunks 25 \
  --astra-max-tool-calls 100 --astra-max-wall-seconds 300 \
  --max-steps 520 --episode-horizon 1000 \
  --replan-steps 5 --gpu-index 0 --reset-retries 12
```

For the previously configured LIBERO-10 Task 8 experiment, explicitly use `--task-ids 8` and `--episode-horizon 1300`. `--max-steps` is the π0.5 phase limit; the episode horizon remains a separate hard cap. On an approved recovery return, only the π0.5 phase counter resets to zero—the episode-wide environment step count does not. If a later π0.5 phase exhausts its budget, the runner can initiate the configured Astra takeover path.

## Action-protocol diagnostic

The diagnostic uses the same production `LiberoEefAdapter` as the live Astra executor. It does not move the robot unless `--motion` is supplied; that option executes small signed motions in a fresh simulator episode.

```bash
python scripts/audit_libero_action_protocol.py \
  --openpi-root "$OPENPI_ROOT" --gpu-index 0 \
  --task-id 0 --output "$OUTPUT_DIR/action_audit.json"
```

Add `--motion` only when a bounded simulated motion check is intended.

## Repository contents

- `models/`: CALL decision/control, handoff contracts, checkpoint metadata, Astra protocol/executor, provider transport, and LIBERO adapter.
- `scripts/run_call_astra_v1_closed_loop.py`: episode/task runner.
- `scripts/audit_libero_action_protocol.py`: production-adapter audit.
- `scripts/probe_astra_*.py`: no-robot provider/transport diagnostics; require separately configured credentials only when used.

The old lab-specific multi-suite shell pipeline is intentionally not included: it embedded machine-specific paths and depended on reporting/training tools outside this deployment runtime.
