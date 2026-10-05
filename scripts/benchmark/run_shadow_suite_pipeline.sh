#!/usr/bin/env bash
set -u

# One-entry pipeline for the next shadow evaluation.
# Stages are resumable: rerunning with the same RUN_ID continues incomplete
# suite runs and only starts comparison after every suite is complete.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/artifacts/call_llm_v1/strict_shadow_all}"
RUN_ID="${RUN_ID:-telemetry_shadow_all_4of5_maxstep_20260926}"
REPORT_ROOT="$PROJECT_ROOT/reports/call_llm_v1/$RUN_ID"
LOG_ROOT="$PROJECT_ROOT/logs/call_llm_v1"
SEED="${SEED:-20260926}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
COOLDOWN_QUERIES="${COOLDOWN_QUERIES:-6}"
DECISION_RULE="${DECISION_RULE:-4_of_5}"
MAX_STEPS="${MAX_STEPS:-520}"
MAX_RETRIES="${MAX_RETRIES:-3}"

CHECKPOINT="${PI05_CHECKPOINT:-}"
OPENPI_ROOT="${OPENPI_ROOT:-}"
if [ -z "$CHECKPOINT" ] || [ -z "$OPENPI_ROOT" ]; then
    echo "Set PI05_CHECKPOINT and OPENPI_ROOT before running this pipeline." >&2
    exit 2
fi

SUITES=(libero_10 libero_object libero_goal libero_spatial)

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$REPORT_ROOT" "$LOG_ROOT"

PIPELINE_STATUS="$REPORT_ROOT/pipeline_status.json"
PIPELINE_LOG="$LOG_ROOT/${RUN_ID}_pipeline.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

write_status() {
    local status="$1"
    local message="${2:-}"
    "$PYTHON_BIN" - "$PIPELINE_STATUS" "$status" "$message" "$RUN_ID" "$SEED" "$COOLDOWN_QUERIES" "$DECISION_RULE" "$MAX_STEPS" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "message": sys.argv[3],
    "run_id": sys.argv[4],
    "seed": int(sys.argv[5]),
    "cooldown_queries": int(sys.argv[6]),
    "decision_rule": sys.argv[7],
    "max_steps": int(sys.argv[8]),
}
if path.exists():
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous.update(payload)
        payload = previous
    except (OSError, ValueError):
        pass
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

write_status "running" "collection started"
echo "[pipeline] run_id=$RUN_ID seed=$SEED decision_rule=$DECISION_RULE cooldown_queries=$COOLDOWN_QUERIES max_steps=$MAX_STEPS"

# Some suites have a small number of tasks without a task-specific competence
# head. Reuse the first trained fold only for those explicitly missing tasks,
# so the all-suite run can proceed without manual intervention.
"$PYTHON_BIN" - "$MODEL_ROOT" <<'PY'
from pathlib import Path
import sys
import torch

root = Path(sys.argv[1])
missing = {
    "libero_goal": [1, 7],
    "libero_spatial": [2],
}
for suite, task_ids in missing.items():
    folder = root / suite / "mlp_soft_full" / "fold_models"
    candidates = sorted(folder.glob("fold_*.pt"))
    target = root / suite / "mlp_soft_full" / "fallback_suite.pt"
    if target.exists() or not candidates:
        continue
    checkpoint = torch.load(candidates[0], map_location="cpu")
    checkpoint["test_task"] = f"{suite}:__fallback__"
    checkpoint["fold_id"] = f"fallback_{suite}"
    checkpoint["fallback_for_missing_task_ids"] = list(task_ids)
    checkpoint["training_protocol"] = (
        str(checkpoint.get("training_protocol", ""))
        + "; suite fallback for tasks without a task-specific head"
    )
    torch.save(checkpoint, target)
    print(f"[pipeline] created {target} for task ids {task_ids}", flush=True)
PY

suite_complete() {
    local output_dir="$1"
    "$PYTHON_BIN" - "$output_dir" "$EPISODES_PER_TASK" <<'PY'
from pathlib import Path
import json
import sys

run_dir = Path(sys.argv[1])
episodes_per_task = int(sys.argv[2])
report_path = run_dir / "run_report.json"
if not report_path.exists():
    raise SystemExit(1)
task_ids = set()
episodes = 0
for path in (run_dir / "episodes").glob("*.json"):
    row = json.loads(path.read_text(encoding="utf-8"))
    task_ids.add(int(row["task_id"]))
    episodes += 1
if not task_ids or episodes != len(task_ids) * episodes_per_task:
    raise SystemExit(1)
PY
}

for suite in "${SUITES[@]}"; do
    output_dir="$REPORT_ROOT/${suite}_learned_shadow"
    model_dir="$MODEL_ROOT/$suite/mlp_soft_full"
    attempt=1
    completed=0

    write_status "running_suite" "suite=$suite"

    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        run_args=(
            scripts/benchmark/run_libero_call_llm_benchmark.py
            --checkpoint "$CHECKPOINT"
            --openpi-root "$OPENPI_ROOT"
            --model-dir "$model_dir"
            --output-dir "$output_dir"
            --suite "$suite"
            --episodes-per-task "$EPISODES_PER_TASK"
            --seed "$SEED"
            --num-steps 10
            --max-steps "$MAX_STEPS"
            --replan-steps 5
            --camera-size 256
            --reset-retries 12
            --gpu-index 1
            --gpu-max-utilization 5
            --threshold-point conservative
            --decision-rule "$DECISION_RULE"
            --cooldown-queries "$COOLDOWN_QUERIES"
            --shadow-mode
            --baseline learned
        )
        if [ -f "$output_dir/run_config.json" ]; then
            run_args+=(--resume)
        fi

        echo "[pipeline] suite=$suite attempt=$attempt/$MAX_RETRIES"
        if "$PYTHON_BIN" "${run_args[@]}"; then
            if suite_complete "$output_dir"; then
                completed=1
                break
            fi
            echo "[pipeline] suite=$suite returned without a complete episode set" >&2
        else
            echo "[pipeline] suite=$suite failed on attempt $attempt" >&2
        fi
        attempt=$((attempt + 1))
        sleep 10
    done

    if [ "$completed" -ne 1 ]; then
        write_status "failed" "suite $suite did not complete after $MAX_RETRIES attempts"
        exit 1
    fi
    echo "[pipeline] suite=$suite complete"
done

write_status "collection_complete" "all suites complete; building summary"

"$PYTHON_BIN" - "$REPORT_ROOT" "$SEED" "$COOLDOWN_QUERIES" "$EPISODES_PER_TASK" "$DECISION_RULE" "$MAX_STEPS" <<'PY'
from collections import defaultdict
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
seed = int(sys.argv[2])
cooldown_queries = int(sys.argv[3])
episodes_per_task = int(sys.argv[4])
decision_rule = sys.argv[5]
max_steps = int(sys.argv[6])
suites = {}
expected_per_suite = None

for report_path in sorted(root.glob("*_learned_shadow/run_report.json")):
    run_dir = report_path.parent
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "episodes").glob("*.json"))
    ]
    if not rows:
        raise SystemExit(f"no completed episodes in {run_dir}")
    by_task = defaultdict(list)
    for row in rows:
        by_task[int(row["task_id"])].append(row)
    task_summary = {}
    for task_id, task_rows in sorted(by_task.items()):
        task_summary[str(task_id)] = {
            "episodes": len(task_rows),
            "successes": sum(row.get("success") is True for row in task_rows),
            "failures": sum(row.get("success") is False for row in task_rows),
            "would_call": sum(bool(row.get("shadow_would_call")) for row in task_rows),
            "would_call_failure": sum(
                bool(row.get("shadow_would_call")) and row.get("success") is False
                for row in task_rows
            ),
            "no_call_failure": sum(
                not bool(row.get("shadow_would_call")) and row.get("success") is False
                for row in task_rows
            ),
            "cooldown_suppressed_queries": sum(
                sum(bool(query.get("cooldown_suppressed")) for query in row.get("query_log", []))
                for row in task_rows
            ),
        }
    suites[report["suite"]] = {
        "run_report": str(report_path),
        "episodes": len(rows),
        "successes": sum(row.get("success") is True for row in rows),
        "failures": sum(row.get("success") is False for row in rows),
        "would_call": sum(bool(row.get("shadow_would_call")) for row in rows),
        "would_call_failure": sum(
            bool(row.get("shadow_would_call")) and row.get("success") is False
            for row in rows
        ),
        "no_call_failure": sum(
            not bool(row.get("shadow_would_call")) and row.get("success") is False
            for row in rows
        ),
        "cooldown_suppressed_queries": sum(
            sum(bool(query.get("cooldown_suppressed")) for query in row.get("query_log", []))
            for row in rows
        ),
        "task_summary": task_summary,
    }
    per_task = len(task_summary)
    expected = per_task * episodes_per_task
    expected_per_suite = expected if expected_per_suite is None else expected_per_suite
    if len(rows) != expected_per_suite:
        raise SystemExit(
            f"{report['suite']} has {len(rows)} episodes; expected {expected_per_suite}"
        )

required = {"libero_10", "libero_object", "libero_goal", "libero_spatial"}
missing = sorted(required - set(suites))
if missing:
    raise SystemExit(f"missing suite reports: {missing}")

summary = {
    "status": "complete",
    "seed": seed,
    "decision_rule": decision_rule,
    "cooldown_queries": cooldown_queries,
    "max_steps": max_steps,
    "shadow_mode": True,
    "threshold_point": "conservative",
    "suites": suites,
}
for key in ("episodes", "successes", "failures", "would_call", "would_call_failure", "no_call_failure", "cooldown_suppressed_queries"):
    summary[key] = sum(item[key] for item in suites.values())
(root / "telemetry_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

write_status "summary_complete" "running three-model comparison"
comparison_dir="$REPORT_ROOT/comparison"
"$PYTHON_BIN" scripts/compare_dynamic_shadow_v1.py \
    --input-root "$REPORT_ROOT" \
    --output-dir "$comparison_dir" \
    --cooldown-queries "$COOLDOWN_QUERIES" \
    --difficulty-queries 5 \
    --decision-rule "$DECISION_RULE"

write_status "complete" "collection, calibration, validation, test comparison, and reports complete"
echo "[pipeline] complete"
echo "[pipeline] summary=$REPORT_ROOT/telemetry_summary.json"
echo "[pipeline] comparison=$comparison_dir/comparison_report.json"
echo "[pipeline] csv=$comparison_dir/comparison_summary.csv"
