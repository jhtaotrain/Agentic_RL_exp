#!/bin/bash
# Sequential experiment runner for RAGEN.
# Runs multiple configs back-to-back under one project name, with distinct
# experiment names. Each run gets its own log file. Ray is cleanly stopped
# between runs so state does not leak.
#
# Usage:
#   bash run_experiments.sh          # run all experiments below
#   bash run_experiments.sh exp_a    # run a single experiment by name
#
# Recommended launch (so it survives SSH disconnects):
#   tmux new -s experiments
#   bash run_experiments.sh 2>&1 | tee run_experiments.log
#   Ctrl+b d   # detach; reattach later with: tmux attach -t experiments

set -u  # undefined vars are errors; do NOT use -e so one failure doesn't abort the queue

PROJECT_NAME="ragen_mechanism5"
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
COMMON_OVERRIDES="agent_proxy.context_window_mode=full actor_rollout_ref.rollout.trajectory_filter.unit=episode seed.train=24681357 seed.val=97531"
COMMON_OVERRIDES="${COMMON_OVERRIDES} trainer.total_training_steps=150"

# ---------------------------------------------------------------------------
# Define experiments here.
# Each entry is: "<experiment_name>|<hydra overrides>"
# The script will launch:
#   python train.py \
#     trainer.project_name=$PROJECT_NAME \
#     trainer.experiment_name=<name> \
#     <overrides>
# ---------------------------------------------------------------------------

EXPERIMENTS=(
  # TODO: fill in your experiments. Examples:
  # "exp_a|actor_rollout_ref.rollout.rollout_filter_ratio=0.25 trainer.total_training_steps=200"
  # "exp_b|actor_rollout_ref.rollout.rollout_filter_ratio=0.5  trainer.total_training_steps=200"
  # "exp_c|algorithm.adv_estimator=grpo actor_rollout_ref.actor.use_kl_loss=True"

  "base_rollout_filter_0.25|\
${COMMON_OVERRIDES} \
actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
actor_rollout_ref.rollout.trajectory_filter.enable=False \
"

  "rollout_0.25_traj_0.25_A_sample_1|\
${COMMON_OVERRIDES} \
actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
actor_rollout_ref.rollout.trajectory_filter.enable=True \
actor_rollout_ref.rollout.trajectory_filter.ratio=0.25 \
actor_rollout_ref.rollout.trajectory_filter.score_type=adv_abs \
actor_rollout_ref.rollout.trajectory_filter.mode=sample \
actor_rollout_ref.rollout.trajectory_filter.alpha=1.0 \
"


  "rollout_0.25_traj_0.25_ASL_sample_1|\
${COMMON_OVERRIDES} \
actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
actor_rollout_ref.rollout.trajectory_filter.enable=True \
actor_rollout_ref.rollout.trajectory_filter.ratio=0.25 \
actor_rollout_ref.rollout.trajectory_filter.score_type=adv_abs_div_sqrt_len \
actor_rollout_ref.rollout.trajectory_filter.mode=sample \
actor_rollout_ref.rollout.trajectory_filter.alpha=1.0 \
"

  "rollout_0.25_traj_0.25_AP_sample_1|\
${COMMON_OVERRIDES} \
actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
actor_rollout_ref.rollout.trajectory_filter.enable=True \
actor_rollout_ref.rollout.trajectory_filter.ratio=0.25 \
actor_rollout_ref.rollout.trajectory_filter.score_type=adv_signed_pos \
actor_rollout_ref.rollout.trajectory_filter.mode=sample \
actor_rollout_ref.rollout.trajectory_filter.alpha=1.0 \
"

#   "rollout_0.25_traj_0.25_ASTD_sample_1|\
# ${COMMON_OVERRIDES} \
# actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
# actor_rollout_ref.rollout.trajectory_filter.enable=True \
# actor_rollout_ref.rollout.trajectory_filter.ratio=0.25 \
# actor_rollout_ref.rollout.trajectory_filter.score_type=adv_std \
# actor_rollout_ref.rollout.trajectory_filter.mode=sample \
# actor_rollout_ref.rollout.trajectory_filter.alpha=1.0 \
# "

  "rollout_0.25_traj_0.5_A_sample_1|\
${COMMON_OVERRIDES} \
actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
actor_rollout_ref.rollout.trajectory_filter.enable=True \
actor_rollout_ref.rollout.trajectory_filter.ratio=0.5 \
actor_rollout_ref.rollout.trajectory_filter.score_type=adv_abs \
actor_rollout_ref.rollout.trajectory_filter.mode=sample \
actor_rollout_ref.rollout.trajectory_filter.alpha=1.0 \
"

#   "rollout_0.25_traj_0.25_A_topk_1|\
# ${COMMON_OVERRIDES} \
# actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
# actor_rollout_ref.rollout.trajectory_filter.enable=True \
# actor_rollout_ref.rollout.trajectory_filter.ratio=0.25 \
# actor_rollout_ref.rollout.trajectory_filter.score_type=adv_abs \
# actor_rollout_ref.rollout.trajectory_filter.mode=topk \
# actor_rollout_ref.rollout.trajectory_filter.alpha=1.0 \
# "

  "rollout_0.25_traj_0.25_A_sample_2|\
${COMMON_OVERRIDES} \
actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
actor_rollout_ref.rollout.trajectory_filter.enable=True \
actor_rollout_ref.rollout.trajectory_filter.ratio=0.25 \
actor_rollout_ref.rollout.trajectory_filter.score_type=adv_abs \
actor_rollout_ref.rollout.trajectory_filter.mode=sample \
actor_rollout_ref.rollout.trajectory_filter.alpha=2.0 \
"

  "rollout_0.25_traj_0.25_AL_sample_1|\
${COMMON_OVERRIDES} \
actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
actor_rollout_ref.rollout.trajectory_filter.enable=True \
actor_rollout_ref.rollout.trajectory_filter.ratio=0.25 \
actor_rollout_ref.rollout.trajectory_filter.score_type=adv_abs_x_length \
actor_rollout_ref.rollout.trajectory_filter.mode=sample \
actor_rollout_ref.rollout.trajectory_filter.alpha=1.0 \
"

#   "rollout_0.25_traj_0.5_AL_sample_1|\
# ${COMMON_OVERRIDES} \
# actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
# actor_rollout_ref.rollout.trajectory_filter.enable=True \
# actor_rollout_ref.rollout.trajectory_filter.ratio=0.5 \
# actor_rollout_ref.rollout.trajectory_filter.score_type=adv_abs_x_length \
# actor_rollout_ref.rollout.trajectory_filter.mode=sample \
# actor_rollout_ref.rollout.trajectory_filter.alpha=1.0 \
# "

#   "rollout_0.25_traj_0.25_AL_topk_1|\
# ${COMMON_OVERRIDES} \
# actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
# actor_rollout_ref.rollout.trajectory_filter.enable=True \
# actor_rollout_ref.rollout.trajectory_filter.ratio=0.25 \
# actor_rollout_ref.rollout.trajectory_filter.score_type=adv_abs_x_length \
# actor_rollout_ref.rollout.trajectory_filter.mode=topk \
# actor_rollout_ref.rollout.trajectory_filter.alpha=1.0 \
# "

#   "rollout_0.25_traj_0.25_AL_sample_2|\
# ${COMMON_OVERRIDES} \
# actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
# actor_rollout_ref.rollout.trajectory_filter.enable=True \
# actor_rollout_ref.rollout.trajectory_filter.ratio=0.25 \
# actor_rollout_ref.rollout.trajectory_filter.score_type=adv_abs_x_length \
# actor_rollout_ref.rollout.trajectory_filter.mode=sample \
# actor_rollout_ref.rollout.trajectory_filter.alpha=2.0 \
# "

)

run_one() {
  local name="$1"
  local overrides="$2"
  local log_file="$LOG_DIR/${name}.log"

  echo "================================================================"
  echo "[$(date -Is)] START  experiment=$name"
  echo "  overrides: $overrides"
  echo "  log:       $log_file"
  echo "================================================================"

  # Make sure no stale Ray cluster from a prior run/crash is lingering.
  ray stop --force >/dev/null 2>&1 || true

  # shellcheck disable=SC2086
  python train.py \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$name" \
    $overrides \
    > "$log_file" 2>&1

  local status=$?

  # Clean up Ray state regardless of outcome so the next run starts fresh.
  ray stop --force >/dev/null 2>&1 || true

  if [[ $status -eq 0 ]]; then
    echo "[$(date -Is)] DONE   experiment=$name (exit 0)"
  else
    echo "[$(date -Is)] FAILED experiment=$name (exit $status) — see $log_file"
  fi
  echo
  return $status
}

# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------
FILTER="${1:-}"
failed=()

for entry in "${EXPERIMENTS[@]}"; do
  name="${entry%%|*}"
  overrides="${entry#*|}"

  if [[ -n "$FILTER" && "$name" != "$FILTER" ]]; then
    continue
  fi

  run_one "$name" "$overrides" || failed+=("$name")
done

echo "================================================================"
echo "[$(date -Is)] ALL DONE"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed experiments: ${failed[*]}"
  exit 1
fi
echo "All experiments succeeded."
