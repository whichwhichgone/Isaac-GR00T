import torch


SPLIT_ACTION_LOSS_KEYS = (
    "selected_loss",
    "remaining_loss",
)


def build_selected_action_dim_mask(
    selected_action_dims: list[int] | None, action_dim: int
) -> torch.Tensor | None:
    """Build a validated mask for arbitrary zero-based action dimensions."""
    if selected_action_dims is None:
        return None
    if not selected_action_dims:
        raise ValueError("selected_action_dims must contain at least one action dimension")
    if len(set(selected_action_dims)) != len(selected_action_dims):
        raise ValueError(f"selected_action_dims contains duplicate indices: {selected_action_dims}")

    invalid_dims = [dim for dim in selected_action_dims if not 0 <= dim < action_dim]
    if invalid_dims:
        raise ValueError(
            f"selected_action_dims must be in [0, {action_dim - 1}], "
            f"got invalid indices {invalid_dims}"
        )

    selected_dim_mask = torch.zeros(action_dim, dtype=torch.bool)
    selected_dim_mask[selected_action_dims] = True
    return selected_dim_mask


def compute_weighted_action_loss(
    action_loss: torch.Tensor,
    action_mask: torch.Tensor,
    selected_dim_mask: torch.Tensor | None,
    selected_action_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute total loss and detached selected/remaining batch losses."""
    if selected_dim_mask is None:
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        return loss, {}

    selected_loss_sum = action_loss[..., selected_dim_mask].sum()
    remaining_loss_sum = action_loss[..., ~selected_dim_mask].sum()
    selected_loss_count = action_mask[..., selected_dim_mask].sum()
    remaining_loss_count = action_mask[..., ~selected_dim_mask].sum()

    selected_loss = selected_loss_sum / selected_loss_count.clamp_min(1e-6)
    remaining_loss = remaining_loss_sum / remaining_loss_count.clamp_min(1e-6)
    loss = remaining_loss + selected_action_weight * selected_loss

    return loss, {
        "selected_loss": selected_loss.detach(),
        "remaining_loss": remaining_loss.detach(),
    }


def aggregate_action_loss_stats(gathered_values: torch.Tensor) -> dict[str, float]:
    """Average selected/remaining batch losses across steps and processes."""
    expected_width = len(SPLIT_ACTION_LOSS_KEYS) + 1
    if gathered_values.ndim != 2 or gathered_values.shape[1] != expected_width:
        raise ValueError(
            f"gathered_values must have shape (num_processes, {expected_width}), "
            f"got {tuple(gathered_values.shape)}"
        )

    total_count = gathered_values[:, 2].sum().clamp_min(1.0)
    return {
        "selected_loss": (gathered_values[:, 0].sum() / total_count).item(),
        "remaining_loss": (gathered_values[:, 1].sum() / total_count).item(),
    }
