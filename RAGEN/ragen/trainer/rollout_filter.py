"""Utilities for filtering rollout trajectories before PPO updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from verl import DataProto


@dataclass
class RolloutFilterConfig:
    """Configuration container for rollout filtering."""

    ratio: float
    filter_type: str
    group_size: int
    num_groups: int
    metric: str = "reward_variance"


@dataclass
class TrajectoryFilterConfig:
    """Configuration container for post-advantage trajectory resampling."""

    enable: bool = False
    ratio: float = 1.0
    score_type: str = "adv_abs"
    mode: str = "topk"
    alpha: float = 1.0


class RolloutFilter:
    """Base class for rollout filters."""

    def __init__(self, config: RolloutFilterConfig):
        self.config = config

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    @property
    def ratio(self) -> float:
        return self.config.ratio

    @property
    def filter_type(self) -> str:
        return self.config.filter_type

    @property
    def group_size(self) -> int:
        return self.config.group_size

    @property
    def num_groups(self) -> int:
        return self.config.num_groups

    def _select_top_groups(self, scores: torch.Tensor) -> torch.Tensor:
        rollout_filter_ratio = self.ratio
        if rollout_filter_ratio >= 1:
            return torch.arange(self.num_groups, device=scores.device)

        k = max(int(rollout_filter_ratio * self.num_groups), 1)

        if self.filter_type == "smallest":
            top_groups = (-scores).topk(k).indices
        elif self.filter_type == "largest":
            top_groups = scores.topk(k).indices
        else:
            raise ValueError(f"Invalid rollout filter type: {self.filter_type}")

        return top_groups

    def _groups_to_mask(self, top_groups: torch.Tensor, group_size: int = None) -> torch.Tensor:
        device = top_groups.device
        if group_size is None:
            group_size = self.group_size
        mask = torch.zeros(self.num_groups, dtype=torch.bool, device=device)
        if top_groups.numel() > 0:
            mask[top_groups] = True
        mask = mask.unsqueeze(1).expand(-1, group_size).reshape(-1).cpu()
        return mask

    def _apply_mask(self, batch: DataProto, mask: torch.Tensor) -> DataProto:
        batch.batch = batch.batch[mask]

        if batch.non_tensor_batch is not None:
            np_mask = mask.cpu().numpy()
            for key, value in batch.non_tensor_batch.items():
                if isinstance(value, np.ndarray):
                    batch.non_tensor_batch[key] = value[np_mask]
                else:
                    batch.non_tensor_batch[key] = [v for v, m in zip(value, np_mask) if m]

        return batch

    def _build_base_metrics(
        self,
        in_group_std: torch.Tensor,
        in_group_max: torch.Tensor,
        in_group_mean: torch.Tensor,
        top_groups: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        metrics = {
            "rollout/in_group_std": in_group_std.mean(),
            "rollout/in_group_max": in_group_max.mean(),
            "rollout/in_group_mean": in_group_mean.mean(),
        }

        chosen = top_groups
        metrics.update(
            {
                "rollout/chosen_in_group_std": in_group_std[chosen].mean(),
                "rollout/chosen_in_group_max": in_group_max[chosen].mean(),
                "rollout/chosen_in_group_mean": in_group_mean[chosen].mean(),
            }
        )
        return metrics


class NullRolloutFilter(RolloutFilter):
    """No-op rollout filter used when filtering is disabled."""

    def __init__(self) -> None:
        super().__init__(
            RolloutFilterConfig(
                ratio=1.0,
                filter_type="largest",
                group_size=1,
                num_groups=1,
                metric="reward_variance",
            )
        )

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, torch.Tensor]]:
        return batch, {}


class RewardRolloutFilter(RolloutFilter):
    """Filters rollouts based on reward statistics within groups."""

    _METRIC_OPTIONS = {"reward", "reward_variance"}

    def __init__(self, config: RolloutFilterConfig) -> None:
        super().__init__(config)
        if config.metric not in self._METRIC_OPTIONS:
            raise ValueError(
                f"RewardRolloutFilter only supports metrics {self._METRIC_OPTIONS}, got {config.metric}"
            )

    def _selection_scores(
        self, in_group_std: torch.Tensor, in_group_mean: torch.Tensor
    ) -> torch.Tensor:
        if self.config.metric == "reward":
            return in_group_mean
        return in_group_std

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, torch.Tensor]]:
        rollout_filter_ratio = self.ratio
        num_groups = self.num_groups

        # Check if this is turn-level mode (single_turn/limited_multi_turn, indicated by episode_ids)
        has_episode_ids = (
            batch.non_tensor_batch is not None
            and "episode_ids" in batch.non_tensor_batch
        )

        if has_episode_ids:
            # Turn-level mode: aggregate by episode first
            episode_ids = batch.non_tensor_batch["episode_ids"]
            group_ids = batch.non_tensor_batch["group_ids"]
            all_scores = batch.batch["original_rm_scores"].sum(dim=-1)

            # Get unique episodes and their rewards
            unique_episodes = []
            episode_to_first_idx = {}
            for i, eid in enumerate(episode_ids):
                if eid not in episode_to_first_idx:
                    unique_episodes.append(eid)
                    episode_to_first_idx[eid] = i

            # Get episode-level rewards and group_ids
            num_episodes = len(unique_episodes)
            episode_rewards = torch.zeros(num_episodes, device=all_scores.device)
            episode_group_ids = []
            for i, eid in enumerate(unique_episodes):
                idx = episode_to_first_idx[eid]
                episode_rewards[i] = all_scores[idx]
                episode_group_ids.append(group_ids[idx])

            # Calculate group_size as episodes per group
            group_size = num_episodes // num_groups
            
            if num_episodes % num_groups != 0:
                raise ValueError(
                    f"Number of episodes ({num_episodes}) must be divisible by num_groups ({num_groups})"
                )
            
            # Reshape to (num_groups, group_size)
            rm_scores = episode_rewards.view(num_groups, group_size)
        else:
            # Original mode: each sample is an episode
            actual_batch_size = batch.batch["original_rm_scores"].shape[0]
            group_size = actual_batch_size // num_groups
            rm_scores = batch.batch["original_rm_scores"].sum(dim=-1).view(num_groups, group_size)

        in_group_std = rm_scores.std(dim=-1)
        in_group_max = rm_scores.max(dim=-1).values
        in_group_mean = rm_scores.mean(dim=-1)

        selection_scores = self._selection_scores(in_group_std, in_group_mean)
        top_groups = self._select_top_groups(selection_scores)

        metrics = self._build_base_metrics(in_group_std, in_group_max, in_group_mean, top_groups)
        metrics.update(
            {
                "rollout/in_group_reward_std": in_group_std.mean(),
                "rollout/in_group_reward_max": in_group_max.mean(),
                "rollout/in_group_reward_mean": in_group_mean.mean(),
                "rollout/chosen_in_group_reward_std": in_group_std[top_groups].mean(),
                "rollout/chosen_in_group_reward_max": in_group_max[top_groups].mean(),
                "rollout/chosen_in_group_reward_mean": in_group_mean[top_groups].mean(),
            }
        )

        if rollout_filter_ratio >= 1:
            return batch, metrics

        if has_episode_ids:
            # Build mask for turn-level samples based on selected groups
            # First, find which episodes belong to selected groups
            selected_episodes = set()
            for gid in top_groups.cpu().tolist():
                start_ep = gid * group_size
                end_ep = start_ep + group_size
                for ep_idx in range(start_ep, end_ep):
                    selected_episodes.add(unique_episodes[ep_idx])

            # Build turn-level mask
            mask = torch.tensor(
                [episode_ids[i] in selected_episodes for i in range(len(episode_ids))],
                dtype=torch.bool
            )
        else:
            mask = self._groups_to_mask(top_groups, group_size)

        batch = self._apply_mask(batch, mask)

        return batch, metrics


class EntropyRolloutFilter(RolloutFilter):
    """Filters rollouts based on policy entropy statistics within groups."""

    _METRIC_OPTIONS = {"entropy", "entropy_variance"}

    def __init__(
        self,
        config: RolloutFilterConfig,
        compute_log_prob: Callable[[DataProto], DataProto],
    ) -> None:
        super().__init__(config)
        if config.metric not in self._METRIC_OPTIONS:
            raise ValueError(
                f"EntropyRolloutFilter only supports metrics {self._METRIC_OPTIONS}, got {config.metric}"
            )
        self._compute_log_prob = compute_log_prob

    def _selection_scores(
        self, in_group_std: torch.Tensor, in_group_mean: torch.Tensor
    ) -> torch.Tensor:
        if self.config.metric == "entropy":
            return in_group_mean
        return in_group_std

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, torch.Tensor]]:
        rollout_filter_ratio = self.ratio
        num_groups = self.num_groups

        if "entropys" not in batch.batch:
            log_prob = self._compute_log_prob(batch)
            batch = batch.union(log_prob)

        entropys = batch.batch["entropys"]
        loss_mask = batch.batch.get("loss_mask")
        if loss_mask is None:
            loss_mask = batch.batch.get("response_mask")
        if loss_mask is None:
            raise ValueError("EntropyRolloutFilter requires loss_mask or response_mask in the batch")

        loss_mask = loss_mask.to(entropys.device)
        token_counts = loss_mask.sum(dim=-1).clamp(min=1)
        entropy_per_traj = (entropys * loss_mask).sum(dim=-1) / token_counts

        # Check if this is turn-level mode (single_turn/limited_multi_turn, indicated by episode_ids)
        has_episode_ids = (
            batch.non_tensor_batch is not None
            and "episode_ids" in batch.non_tensor_batch
        )

        if has_episode_ids:
            # Turn-level mode: aggregate by episode first
            episode_ids = batch.non_tensor_batch["episode_ids"]

            # Get unique episodes and their entropy (average across turns)
            unique_episodes = []
            episode_to_indices = {}
            for i, eid in enumerate(episode_ids):
                if eid not in episode_to_indices:
                    unique_episodes.append(eid)
                    episode_to_indices[eid] = []
                episode_to_indices[eid].append(i)

            # Get episode-level entropy (mean of all turns)
            num_episodes = len(unique_episodes)
            episode_entropy = torch.zeros(num_episodes, device=entropy_per_traj.device)
            for i, eid in enumerate(unique_episodes):
                indices = episode_to_indices[eid]
                episode_entropy[i] = entropy_per_traj[indices].mean()

            # Calculate group_size as episodes per group
            group_size = num_episodes // num_groups

            if num_episodes % num_groups != 0:
                raise ValueError(
                    f"Number of episodes ({num_episodes}) must be divisible by num_groups ({num_groups})"
                )

            # Reshape to (num_groups, group_size)
            entropy_per_group = episode_entropy.view(num_groups, group_size)
        else:
            # Original mode: each sample is an episode
            actual_batch_size = entropy_per_traj.shape[0]
            group_size = actual_batch_size // num_groups
            entropy_per_group = entropy_per_traj.view(num_groups, group_size)

        in_group_std = entropy_per_group.std(dim=-1)
        in_group_max = entropy_per_group.max(dim=-1).values
        in_group_mean = entropy_per_group.mean(dim=-1)

        selection_scores = self._selection_scores(in_group_std, in_group_mean)
        top_groups = self._select_top_groups(selection_scores)

        metrics = self._build_base_metrics(in_group_std, in_group_max, in_group_mean, top_groups)
        metrics.update(
            {
                "rollout/in_group_entropy_std": in_group_std.mean(),
                "rollout/in_group_entropy_max": in_group_max.mean(),
                "rollout/in_group_entropy_mean": in_group_mean.mean(),
                "rollout/chosen_in_group_entropy_std": in_group_std[top_groups].mean(),
                "rollout/chosen_in_group_entropy_max": in_group_max[top_groups].mean(),
                "rollout/chosen_in_group_entropy_mean": in_group_mean[top_groups].mean(),
            }
        )

        if rollout_filter_ratio >= 1:
            return batch, metrics

        if has_episode_ids:
            # Build mask for turn-level samples based on selected groups
            selected_episodes = set()
            for gid in top_groups.cpu().tolist():
                start_ep = gid * group_size
                end_ep = start_ep + group_size
                for ep_idx in range(start_ep, end_ep):
                    selected_episodes.add(unique_episodes[ep_idx])

            # Build turn-level mask
            mask = torch.tensor(
                [episode_ids[i] in selected_episodes for i in range(len(episode_ids))],
                dtype=torch.bool
            )
        else:
            mask = self._groups_to_mask(top_groups, group_size)

        batch = self._apply_mask(batch, mask)

        return batch, metrics


class TrajectoryRolloutFilter:
    """Resamples trajectories after advantages have been computed.

    The trajectory filter first selects a subset of trajectories or episodes
    according to the configured score and selection mode. It then restores the
    original trajectory volume by resampling from the retained support with
    score-based weights. This keeps the original RAGEN rollout filter
    untouched while turning the new trajectory-level stage into a fixed-volume
    redistribution mechanism instead of a pure hard filter.
    """

    _SCORE_OPTIONS = {"adv_abs", "adv_abs_x_length"}
    _MODE_OPTIONS = {"topk", "sample"}

    def __init__(self, config: TrajectoryFilterConfig) -> None:
        self.config = config
        if config.score_type not in self._SCORE_OPTIONS:
            raise ValueError(
                f"TrajectoryRolloutFilter only supports score types {self._SCORE_OPTIONS}, got {config.score_type}"
            )
        if config.mode not in self._MODE_OPTIONS:
            raise ValueError(
                f"TrajectoryRolloutFilter only supports modes {self._MODE_OPTIONS}, got {config.mode}"
            )

    @property
    def enable(self) -> bool:
        return self.config.enable

    @property
    def ratio(self) -> float:
        return self.config.ratio

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, torch.Tensor]]:
        if not self.enable or self.ratio >= 1:
            return batch, {}

        if "advantages" not in batch.batch:
            raise ValueError("TrajectoryRolloutFilter requires advantages in the batch")

        mask_key = "response_mask" if "response_mask" in batch.batch else "loss_mask"
        if mask_key not in batch.batch:
            raise ValueError("TrajectoryRolloutFilter requires response_mask or loss_mask in the batch")

        advantages = batch.batch["advantages"]
        valid_mask = batch.batch[mask_key].to(advantages.device)
        signed_advantages = advantages * valid_mask
        abs_advantages = advantages.abs() * valid_mask
        response_lengths = valid_mask.sum(dim=-1).clamp(min=1)
        mean_signed_adv = signed_advantages.sum(dim=-1) / response_lengths
        mean_abs_adv = abs_advantages.sum(dim=-1) / response_lengths
        sample_rewards = None
        if "token_level_scores" in batch.batch:
            sample_rewards = (batch.batch["token_level_scores"] * valid_mask).sum(dim=-1)
        elif "token_level_rewards" in batch.batch:
            sample_rewards = (batch.batch["token_level_rewards"] * valid_mask).sum(dim=-1)

        if self.config.score_type == "adv_abs":
            sample_scores = mean_abs_adv
        elif self.config.score_type == "adv_abs_x_length":
            sample_scores = mean_abs_adv * response_lengths
        else:
            raise ValueError(f"Unsupported trajectory score type: {self.config.score_type}")

        if (
            batch.non_tensor_batch is not None
            and "episode_ids" in batch.non_tensor_batch
        ):
            selected_indices, metrics = self._filter_turn_level(
                batch, sample_scores, response_lengths, mean_signed_adv, sample_rewards
            )
        else:
            selected_indices, metrics = self._filter_episode_level(
                sample_scores, response_lengths, mean_signed_adv, sample_rewards
            )

        batch = self._apply_indices(batch, selected_indices)

        return batch, metrics

    def _apply_indices(self, batch: DataProto, indices: torch.Tensor) -> DataProto:
        if batch.batch is not None:
            batch.batch = batch.batch[indices]

        if batch.non_tensor_batch is not None:
            np_indices = indices.cpu().numpy()
            for key, value in batch.non_tensor_batch.items():
                if isinstance(value, np.ndarray):
                    batch.non_tensor_batch[key] = value[np_indices]
                else:
                    batch.non_tensor_batch[key] = [value[i] for i in np_indices.tolist()]

        return batch

    def _select_indices(self, scores: torch.Tensor) -> torch.Tensor:
        num_items = scores.numel()
        keep_count = max(int(num_items * self.ratio), 1)
        if keep_count >= num_items:
            return torch.arange(num_items, device=scores.device)

        if self.config.mode == "topk":
            return scores.topk(keep_count).indices

        weights = scores.clamp(min=1e-12).pow(self.config.alpha)
        total = weights.sum()
        if not torch.isfinite(total) or total <= 0:
            weights = torch.ones_like(weights) / num_items
        else:
            weights = weights / total
        return torch.multinomial(weights, keep_count, replacement=False)

    def _filter_episode_level(
        self,
        scores: torch.Tensor,
        response_lengths: torch.Tensor,
        signed_mean_adv: torch.Tensor,
        sample_rewards: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        selected_indices = self._select_indices(scores)
        resampled_indices = self._resample_selected_support(
            selected_indices,
            scores[selected_indices],
            target_count=scores.numel(),
        )
        metrics = self._build_metrics(
            scores,
            response_lengths,
            signed_mean_adv,
            sample_rewards,
            selected_indices,
            resampled_indices,
            resampled_count=resampled_indices.numel(),
        )
        return resampled_indices.cpu(), metrics

    def _filter_turn_level(
        self,
        batch: DataProto,
        sample_scores: torch.Tensor,
        response_lengths: torch.Tensor,
        signed_mean_adv: torch.Tensor,
        sample_rewards: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        episode_ids = batch.non_tensor_batch["episode_ids"]
        unique_episode_ids = []
        episode_to_indices = {}
        for idx, episode_id in enumerate(episode_ids):
            if episode_id not in episode_to_indices:
                unique_episode_ids.append(episode_id)
                episode_to_indices[episode_id] = []
            episode_to_indices[episode_id].append(idx)

        episode_scores = torch.zeros(len(unique_episode_ids), device=sample_scores.device)
        episode_lengths = torch.zeros(len(unique_episode_ids), device=sample_scores.device)
        episode_signed_mean_adv = torch.zeros(len(unique_episode_ids), device=sample_scores.device)
        episode_rewards = None
        if sample_rewards is not None:
            episode_rewards = torch.zeros(len(unique_episode_ids), device=sample_scores.device)
        for episode_idx, episode_id in enumerate(unique_episode_ids):
            indices = episode_to_indices[episode_id]
            index_tensor = torch.tensor(indices, device=sample_scores.device, dtype=torch.long)
            episode_scores[episode_idx] = sample_scores[index_tensor].mean()
            episode_lengths[episode_idx] = response_lengths[index_tensor].sum()
            episode_signed_mean_adv[episode_idx] = signed_mean_adv[index_tensor].mean()
            if episode_rewards is not None:
                episode_rewards[episode_idx] = sample_rewards[index_tensor].mean()

        selected_episode_indices = self._select_indices(episode_scores)
        resampled_episode_indices = self._resample_selected_support(
            selected_episode_indices,
            episode_scores[selected_episode_indices],
            target_count=len(unique_episode_ids),
        )

        expanded_sample_indices = []
        for episode_idx in resampled_episode_indices.cpu().tolist():
            episode_id = unique_episode_ids[episode_idx]
            expanded_sample_indices.extend(episode_to_indices[episode_id])

        expanded_sample_indices = torch.tensor(expanded_sample_indices, dtype=torch.long)
        metrics = self._build_metrics(
            episode_scores,
            episode_lengths,
            episode_signed_mean_adv,
            episode_rewards,
            selected_episode_indices,
            resampled_episode_indices,
            resampled_count=resampled_episode_indices.numel(),
        )
        return expanded_sample_indices, metrics

    def _resample_selected_support(
        self,
        selected_indices: torch.Tensor,
        selected_scores: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        """Resample a fixed-volume batch from the retained support."""
        if selected_indices.numel() == 0:
            raise ValueError("TrajectoryRolloutFilter cannot resample an empty selection")

        selected_indices = selected_indices.to(dtype=torch.long)
        if selected_indices.numel() >= target_count:
            return selected_indices[:target_count]

        weights = selected_scores.clamp(min=1e-12).pow(self.config.alpha)
        total = weights.sum()
        if not torch.isfinite(total) or total <= 0:
            weights = torch.ones_like(weights) / selected_indices.numel()
        else:
            weights = weights / total

        sampled_positions = torch.multinomial(
            weights,
            num_samples=target_count,
            replacement=True,
        )
        expanded_indices = selected_indices[sampled_positions]
        if expanded_indices.numel() != target_count:
            raise ValueError(
                f"Resampled trajectory count mismatch: {expanded_indices.numel()} != {target_count}"
            )
        return expanded_indices

    def _build_metrics(
        self,
        scores: torch.Tensor,
        lengths: torch.Tensor,
        signed_mean_adv: torch.Tensor,
        rewards: Optional[torch.Tensor],
        selected_indices: torch.Tensor,
        resampled_indices: torch.Tensor,
        resampled_count: int,
    ) -> Dict[str, torch.Tensor]:
        selected_scores = scores[selected_indices]
        selected_lengths = lengths[selected_indices]
        selected_signed_adv = signed_mean_adv[selected_indices]
        keep_fraction = selected_indices.numel() / max(scores.numel(), 1)
        expansion_factor = resampled_count / max(selected_indices.numel(), 1)
        metrics = {
            "traj_filter/score_mean": scores.mean(),
            "traj_filter/score_max": scores.max(),
            "traj_filter/score_min": scores.min(),
            "traj_filter/selected_score_mean": selected_scores.mean(),
            "traj_filter/selected_score_max": selected_scores.max(),
            "traj_filter/selected_length_mean": selected_lengths.float().mean(),
            "traj_filter/selected_pos_adv_frac": (selected_signed_adv > 0).float().mean(),
            "traj_filter/selected_neg_adv_frac": (selected_signed_adv < 0).float().mean(),
            "traj_filter/selected_pos_adv_mean": selected_signed_adv[selected_signed_adv > 0].mean()
            if (selected_signed_adv > 0).any()
            else torch.tensor(0.0, dtype=torch.float32, device=scores.device),
            "traj_filter/selected_neg_adv_mean": selected_signed_adv[selected_signed_adv < 0].mean()
            if (selected_signed_adv < 0).any()
            else torch.tensor(0.0, dtype=torch.float32, device=scores.device),
            "traj_filter/keep_fraction": torch.tensor(keep_fraction, dtype=torch.float32, device=scores.device),
            "traj_filter/resampled_count": torch.tensor(float(resampled_count), dtype=torch.float32, device=scores.device),
            "traj_filter/expansion_factor": torch.tensor(expansion_factor, dtype=torch.float32, device=scores.device),
        }
        if rewards is not None:
            selected_rewards = rewards[selected_indices]
            metrics["traj_filter/selected_reward_mean"] = selected_rewards.mean()
            metrics["traj_filter/selected_reward_std"] = selected_rewards.std() if selected_rewards.numel() > 1 else torch.tensor(0.0, dtype=torch.float32, device=scores.device)

        if resampled_indices.numel() > 0:
            unique_count = torch.unique(resampled_indices).numel()
            metrics["traj_filter/unique_selected_frac"] = torch.tensor(
                unique_count / max(resampled_indices.numel(), 1),
                dtype=torch.float32,
                device=scores.device,
            )
            sample_counts = torch.bincount(resampled_indices.to(dtype=torch.long), minlength=scores.numel())
            metrics["traj_filter/max_resample_count"] = sample_counts.max().to(dtype=torch.float32)

        return metrics


# Backwards compatibility: preserve older class names.
RewardVarianceRolloutFilter = RewardRolloutFilter
EntropyVarianceRolloutFilter = EntropyRolloutFilter


def build_rollout_filter(
    ratio: float,
    filter_type: str,
    num_groups: int,
    group_size: int,
    metric: Optional[str],
    compute_log_prob: Optional[Callable[[DataProto], DataProto]] = None,
) -> RolloutFilter:
    metric = (metric or "reward_variance").lower()
    metric = {
        "reward_std": "reward_variance",
        "entropy_std": "entropy_variance",
    }.get(metric, metric)

    config = RolloutFilterConfig(
        ratio=ratio,
        filter_type=filter_type,
        num_groups=num_groups,
        group_size=group_size,
        metric=metric,
    )

    if ratio >= 1:
        return NullRolloutFilter()

    if metric in {"reward", "reward_variance"}:
        return RewardRolloutFilter(config)
    if metric in {"entropy", "entropy_variance"}:
        if compute_log_prob is None:
            raise ValueError("Entropy filtering requires a compute_log_prob callable")
        return EntropyRolloutFilter(config, compute_log_prob=compute_log_prob)

    raise ValueError(f"Unsupported rollout filter metric: {metric}")


def build_trajectory_filter(
    enable: bool,
    ratio: float,
    score_type: str,
    mode: str,
    alpha: float = 1.0,
) -> TrajectoryRolloutFilter:
    return TrajectoryRolloutFilter(
        TrajectoryFilterConfig(
            enable=enable,
            ratio=ratio,
            score_type=score_type,
            mode=mode,
            alpha=alpha,
        )
    )
