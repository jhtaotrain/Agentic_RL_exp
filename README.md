# Agentic_RL_exp

Agentic RL experiments built on top of RAGEN.

## Trajectory Resampling Extension

This repository adds an experimental post-advantage trajectory resampling stage to the RAGEN training loop.

### Motivation

The default RAGEN pipeline already supports group-level rollout filtering before policy updates. This extension adds a second filtering stage that runs **after** rewards, values, and advantages are computed. The goal is to make trajectory selection depend on an explicit trajectory utility signal.

This is useful for experiments on trajectory-level data selection, variance-aware resampling, and future importance-sampling style methods.

### Current Design

The current implementation keeps the rollout generation stage unchanged and inserts a new optional filter after `compute_advantage(...)`.

The resulting update order is:

1. Generate rollouts
2. Apply the optional original group-level rollout filter
3. Compute rewards, values, and advantages
4. Compute a trajectory score
5. Resample trajectories
6. Adjust batch size and continue PPO/GRPO updates

### Implemented Scores

The first implementation includes two trajectory score options:

- `adv_abs`: masked mean of `|advantage|`
- `adv_abs_x_length`: masked mean of `|advantage|` multiplied by response length

For turn-level training modes (`single_turn` and `limited_multi_turn`), scores are aggregated to the episode level and all turns from each selected episode are kept together.

### Code Structure

The implementation is split across these files:

- [RAGEN/ragen/trainer/rollout_filter.py](/d:/RL/Agentic_RL_exp/RAGEN/ragen/trainer/rollout_filter.py)
  Adds `TrajectoryFilterConfig`, `TrajectoryRolloutFilter`, and `build_trajectory_filter(...)`.
- [RAGEN/ragen/trainer/agent_trainer.py](/d:/RL/Agentic_RL_exp/RAGEN/ragen/trainer/agent_trainer.py)
  Applies the new trajectory filter after advantage computation and before batch adjustment.
- [RAGEN/config/base.yaml](/d:/RL/Agentic_RL_exp/RAGEN/config/base.yaml)
  Adds the new configuration interface.
- [RAGEN/config/base_default.yaml](/d:/RL/Agentic_RL_exp/RAGEN/config/base_default.yaml)
  Adds the same interface to the default variant.

### Configuration Interface

The new interface lives under:

```yaml
actor_rollout_ref:
  rollout:
    trajectory_filter:
      enable: true
      ratio: 0.5
      score_type: adv_abs_x_length   # adv_abs | adv_abs_x_length
      mode: topk                     # topk | sample
      alpha: 1.0
```

Field definitions:

- `enable`: enables or disables post-advantage trajectory resampling
- `ratio`: fraction of trajectories kept after resampling
- `score_type`: score function used to rank or sample trajectories
- `mode`:
  - `topk`: deterministically keep the highest-scoring trajectories
  - `sample`: sample trajectories without replacement according to score-based probabilities
- `alpha`: score sharpening parameter used only when `mode=sample`

### Relationship to the Original RAGEN Filter

The original group-level rollout filter is still available through:

- `actor_rollout_ref.rollout.rollout_filter_ratio`
- `actor_rollout_ref.rollout.rollout_filter_type`
- `actor_rollout_ref.rollout.rollout_filter_metric`

The new trajectory filter is independent from that mechanism.

Practical settings:

- Disable the old group filter:

```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_ratio: 1
```

- Disable the new trajectory filter:

```yaml
actor_rollout_ref:
  rollout:
    trajectory_filter:
      enable: false
```

- Recover the original RAGEN behavior:

```yaml
actor_rollout_ref:
  rollout:
    trajectory_filter:
      enable: false
```

### Example Usage

Use the new filter with top-k trajectory resampling:

```bash
python train.py \
  actor_rollout_ref.rollout.rollout_filter_ratio=1 \
  actor_rollout_ref.rollout.trajectory_filter.enable=true \
  actor_rollout_ref.rollout.trajectory_filter.ratio=0.5 \
  actor_rollout_ref.rollout.trajectory_filter.score_type=adv_abs_x_length \
  actor_rollout_ref.rollout.trajectory_filter.mode=topk
```

Run a minimal smoke test:

```bash
python train.py \
  trainer.total_training_steps=1 \
  trainer.validation_steps=1 \
  trainer.val_before_train=false \
  trainer.test_freq=0 \
  trainer.save_freq=0 \
  micro_batch_size_per_gpu=1 \
  ppo_mini_batch_size=8 \
  actor_rollout_ref.rollout.max_model_len=512 \
  actor_rollout_ref.rollout.response_length=64 \
  es_manager.train.env_groups=1 \
  es_manager.train.group_size=8 \
  es_manager.train.env_configs.tags='[CoordSokoban]' \
  es_manager.train.env_configs.n_groups='[1]' \
  es_manager.val.env_groups=1 \
  es_manager.val.group_size=1 \
  es_manager.val.env_configs.tags='[CoordSokoban]' \
  es_manager.val.env_configs.n_groups='[1]' \
  actor_rollout_ref.rollout.rollout_filter_ratio=1 \
  actor_rollout_ref.rollout.trajectory_filter.enable=true \
  actor_rollout_ref.rollout.trajectory_filter.ratio=0.5 \
  actor_rollout_ref.rollout.trajectory_filter.score_type=adv_abs \
  actor_rollout_ref.rollout.trajectory_filter.mode=topk
```

### Future Extension Points

The current interface is designed to be easy to extend. Planned future directions include:

- gradient-norm trajectory proxies
- importance-weighted resampling
- softer sampling distributions
- trajectory-level stability or variance proxies

### Gradient Logging

The current code also logs gradient-norm statistics across PPO mini-batches inside each update:

- `actor/grad_norm`
- `actor/grad_norm_mean`
- `actor/grad_norm_std`
- `actor/grad_norm_var`
- `critic/grad_norm`
- `critic/grad_norm_mean`
- `critic/grad_norm_std`
- `critic/grad_norm_var`

These are not exact gradient-estimator variance measurements. They are lightweight update-level diagnostics computed from the sequence of mini-batch gradient norms observed during actor and critic updates.

The actor also logs a lightweight gradient-direction diagnostic on trainable LoRA parameters:

- `actor/lora_grad_cosine`
- `actor/lora_grad_cosine_mean`
- `actor/lora_grad_cosine_std`

This metric is computed as the cosine similarity between consecutive mini-batch gradient vectors on LoRA parameters. It is intended as a cheap proxy for directional consistency across mini-batch updates, rather than a full estimator-variance measurement.

These diagnostics are disabled by default and can be enabled from config:

```yaml
actor_rollout_ref:
  actor:
    diagnostics:
      log_grad_norm_stats: true
      log_lora_grad_cosine: true

critic:
  diagnostics:
    log_grad_norm_stats: true
```
