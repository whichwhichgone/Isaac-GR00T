from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Any
import warnings

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy import BasePolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyClient
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tyro


warnings.simplefilter("ignore", category=FutureWarning)

"""
Example commands:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

"""


def compute_mocap_xyz_error_stats(
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    mocap_dim: int = 9,
) -> dict[str, np.ndarray]:
    """Compute xyz error stats after reshaping actions from [T, 99] to [T, 11, 9]."""
    if gt_action_across_time.shape != pred_action_across_time.shape:
        raise ValueError(
            f"gt_action shape {gt_action_across_time.shape} does not match "
            f"pred_action shape {pred_action_across_time.shape}"
        )
    if gt_action_across_time.ndim != 2:
        raise ValueError(f"Expected actions to be 2D [T, D], got {gt_action_across_time.shape}")

    actual_steps, action_dim = gt_action_across_time.shape  #gt_action_across_time.shape (893, 5100)   pred_action_across_time.shape (893, 5100)
    gt_action_across_time_mocap_flat = gt_action_across_time.reshape(actual_steps, 50, -1)[:,:,3:]
    gt_action_across_time_mocap_11x9 = gt_action_across_time_mocap_flat.reshape(actual_steps, 50, 11, mocap_dim)
    gt_action_across_time_mocap_root_xyz = gt_action_across_time_mocap_11x9[:, :, 0, :3]
    gt_action_across_time_mocap_without_root_xyz = gt_action_across_time_mocap_11x9[:, :, 1:, :3]
    pred_action_across_time_mocap_flat = pred_action_across_time.reshape(actual_steps, 50, -1)[:,:,3:]
    pred_action_across_time_mocap_11x9 = pred_action_across_time_mocap_flat.reshape(actual_steps, 50, 11, mocap_dim)
    pred_action_across_time_mocap_root_xyz = pred_action_across_time_mocap_11x9[:, :, 0, :3]
    pred_action_across_time_mocap_without_root_xyz = pred_action_across_time_mocap_11x9[:, :, 1:, :3]


    xyz_error = gt_action_across_time_mocap_11x9[:, :, :, :3] - pred_action_across_time_mocap_11x9[:, :, :, :3]
    root_xyz_error = gt_action_across_time_mocap_root_xyz - pred_action_across_time_mocap_root_xyz
    without_root_xyz_error = gt_action_across_time_mocap_without_root_xyz - pred_action_across_time_mocap_without_root_xyz
    return {
        "mse_curve": np.mean(xyz_error**2, axis=(1, 2)),  # [T, 3], mean over mocap points
        "mae_curve": np.mean(np.abs(xyz_error), axis=(1, 2)),  # [T, 3], mean over mocap points
        "mse_without_root_curve": np.mean(without_root_xyz_error**2, axis=(1, 2)),  # [T, 3], mean over mocap points
        "mae_without_root_curve": np.mean(np.abs(without_root_xyz_error), axis=(1, 2)),  # [T, 3], mean over mocap points
        "mse_root_curve": np.mean(root_xyz_error**2, axis=(1)),  # [T, 3], mean over root points
        "mae_root_curve": np.mean(np.abs(root_xyz_error), axis=(1)),  # [T,3], mean over root points
        "mse_xyz": np.mean(xyz_error**2, axis=(0, 1, 2)),  # [3], mean over time and points
        "mae_xyz": np.mean(np.abs(xyz_error), axis=(0, 1, 2)),  # [3], mean over time and points
    }, gt_action_across_time_mocap_root_xyz[:,0,:], pred_action_across_time_mocap_root_xyz[:,0,:]


def plot_trajectory_results(
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        action_horizon: Action horizon used for inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        # if state_joints_across_time.shape == gt_action_across_time.shape:
        #     ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()
    
    # Create filename with trajectory ID
    save_plot_path = Path(save_plot_path)
    save_path =  f"{save_plot_path}/root_xyz_real.jpeg"
    
    plt.savefig(save_path)

    plt.close()  # Close the figure to free memory


def plot_trajectory_error(
    error_stats: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    This version saves three separate figures:
    1. mocap xyz error
    2. root xyz error
    3. without_root xyz error
    """
    actual_steps = len(error_stats["mse_curve"])

    coord_names = ["x", "y", "z"]

    plot_groups = {
        "mocap": {
            "mse": error_stats["mse_curve"],
            "mae": error_stats["mae_curve"],
            "title": "Mocap xyz error over time, averaged across 11 points",
        },
        "root": {
            "mse": error_stats["mse_root_curve"],
            "mae": error_stats["mae_root_curve"],
            "title": "Root xyz error over time",
        },
        "without_root": {
            "mse": error_stats["mse_without_root_curve"],
            "mae": error_stats["mae_without_root_curve"],
            "title": "Mocap xyz error over time without root, averaged across non-root points",
        },
    }

    save_plot_path = Path(save_plot_path)
    save_plot_path.mkdir(parents=True, exist_ok=True)

    for group_name, group_data in plot_groups.items():
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(12, 8),
            sharex=True,
        )

        fig.suptitle(
            "Trajectory "
            f"{traj_id} - {group_name} | "
            f"State: {', '.join(state_keys)} | "
            f"Action: {', '.join(action_keys)}",
            fontsize=16,
            color="blue",
        )

        mse_curve = group_data["mse"]
        mae_curve = group_data["mae"]

        for coord_idx, coord_name in enumerate(coord_names):
            axes[0].plot(
                mse_curve[:, coord_idx],
                label=f"{group_name}_{coord_name} mse",
            )
            axes[1].plot(
                mae_curve[:, coord_idx],
                label=f"{group_name}_{coord_name} mae",
            )


        axes[0].set_title(f"{group_data['title']} - MSE")
        axes[0].set_ylabel("MSE")

        axes[1].set_title(f"{group_data['title']} - MAE")
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("MAE")

        fig.tight_layout()

        group_save_path =  f"{save_plot_path}/{group_name}.jpeg"


        fig.savefig(group_save_path)
        print(f"figure save in {group_save_path}")

        plt.close(fig)


def parse_observation_gr00t(
    obs: dict[str, Any], modality_configs: dict[str, Any]
) -> dict[str, Any]:
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs[parsed_key]
            # Add batch dimension
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def parse_action_gr00t(action: dict[str, Any]) -> dict[str, Any]:
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def select_mocap_xyz_dims(actions: np.ndarray, mocap_dim: int = 9) -> np.ndarray:
    """Select xyz dimensions from flattened mocap action blocks.

    The mocap action is expected to be flattened as:
    [point_0_dim_0..8, point_1_dim_0..8, ...].
    This keeps only dims 0, 1, 2 from each point.
    """
    if actions.ndim != 2:
        raise ValueError(f"Expected actions to be 2D [T, D], got shape {actions.shape}")
    if actions.shape[1] % mocap_dim != 0:
        raise ValueError(
            f"Action dim {actions.shape[1]} is not divisible by mocap block dim {mocap_dim}"
        )

    num_points = actions.shape[1] // mocap_dim
    xyz_indices = [
        point_idx * mocap_dim + dim_idx
        for point_idx in range(num_points)
        for dim_idx in range(3)
    ]
    return actions[:, xyz_indices]


def evaluate_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    modality_keys: list[str] | None = None,
    steps=300,
    action_horizon=16,
    checkpoint_name=None,
    save_plot_path=None,
    save_action_json_path=None,
    save_gt_action_json_path=None,
):
    # Ensure steps doesn't exceed trajectory length
    traj = loader[traj_id]
    traj_length = len(traj)
    steps = traj_length - 1
    actual_steps = min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )

    pred_action_across_time = []

    # Extract state and action keys separately and sort for consistent order
    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = (
        loader.modality_configs["action"].modality_keys if modality_keys is None else modality_keys
    )

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")
    for step_count in range(0, actual_steps, action_horizon):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag, allow_padding=True)
        logging.info(f"inferencing at step: {step_count}")
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v  # (T, D)
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        _action_chunk, _ = policy.get_action(parsed_obs)
        action_chunk = parse_action_gr00t(_action_chunk)
        for j in range(action_horizon):
            # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
            # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
            concat_pred_action = np.concatenate(
                [
                    np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                    for key in action_keys
                ],
                axis=0,
            )
            pred_action_across_time.append(concat_pred_action)

    def extract_state_joints(traj: pd.DataFrame, columns: list[str]):
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    # plot the joints
    state_joints_across_time = extract_state_joints(traj, [f"state.{key}" for key in state_keys])
    gt_action_across_time = extract_state_joints(traj, [f"action.{key}" for key in action_keys])[
        :actual_steps
    ]
    pred_action_across_time = np.array(pred_action_across_time)[:actual_steps]

    # Save predicted actions to JSON: list of [root_x, root_y, root_z, qw, qx, qy, qz, joint_0, ..., joint_28]
    json_path = Path(f"/liujinxin/liyifan/Isaac-GR00T/tmp/open_loop_eval/{checkpoint_name}/traj_{traj_id}/pred_action.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(pred_action_across_time.tolist(), f)
    logging.info(f"Saved predicted actions to {json_path}")

    # Save ground truth actions to JSON
    gt_json_path = Path(f"/liujinxin/liyifan/Isaac-GR00T/tmp/open_loop_eval/{checkpoint_name}/traj_{traj_id}/gt_action.json")
    gt_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_json_path, "w") as f:
        json.dump(gt_action_across_time.tolist(), f)
    logging.info(f"Saved ground truth actions to {gt_json_path}")

    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    # calc MSE and MAE for x/y/z, averaging over the 11 mocap points.
    error_stats, gt_action_across_time_mocap_root_xyz, pred_action_across_time_mocap_root_xyz = compute_mocap_xyz_error_stats(gt_action_across_time, pred_action_across_time)
    mse_xyz = error_stats["mse_xyz"]
    mae_xyz = error_stats["mae_xyz"]
    mse = float(np.mean(mse_xyz))
    mae = float(np.mean(mae_xyz))
    logging.info(
        "Unnormalized Action XYZ MSE across single traj "
        f"(x={mse_xyz[0]}, y={mse_xyz[1]}, z={mse_xyz[2]}, mean={mse})"
    )
    logging.info(
        "Unnormalized Action XYZ MAE across single traj "
        f"(x={mae_xyz[0]}, y={mae_xyz[1]}, z={mae_xyz[2]}, mean={mae})"
    )

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")
    logging.info(f"mse_xyz_curve vs time {error_stats['mse_curve'].shape}")
    logging.info(f"mae_xyz_curve vs time {error_stats['mae_curve'].shape}")

    # Plot trajectory results
    plot_trajectory_error(
        error_stats=error_stats,
        traj_id=traj_id,
        state_keys=state_keys,
        action_keys=action_keys,
        save_plot_path= f"/liujinxin/liyifan/Isaac-GR00T/tmp/open_loop_eval/{checkpoint_name}/traj_{traj_id}",
        # save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
    )
    plot_trajectory_results(
        gt_action_across_time = gt_action_across_time_mocap_root_xyz,
        pred_action_across_time = pred_action_across_time_mocap_root_xyz,
        traj_id=traj_id,
        save_plot_path= f"/liujinxin/liyifan/Isaac-GR00T/tmp/open_loop_eval/{checkpoint_name}/traj_{traj_id}"
    )
    return mse, mae


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 2000
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    traj_ids: list[int] = field(default_factory=lambda: [0,1,2,3,4])
    """List of trajectory IDs to evaluate."""

    action_horizon: int = 1
    """Action horizon to evaluate."""

    dataset_path: str = "/liujinxin/liyifan/Isaac-GR00T/dataset/G1_real_6D_window_cont_rel"
    """Path to the dataset."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.UNITREE_G1_29DOF
    """Embodiment tag to use."""

    model_path: str | None = "/liujinxin/liyifan/Isaac-GR00T/checkpoints/G1_real_6D_window_cont_rel/checkpoint-40000"
    """Path to the model checkpoint."""

    checkpoint_name: str = "G1_real_6D_window_cont_rel-40000"
    """Name of the checkpoint to use."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    modality_keys: list[str] | None = None
    """List of modality keys to plot. If None, plot all keys."""

    save_action_json_path: str | None = "./pred_action.json"
    """Path to save predicted actions as JSON. Each timestep is saved as [root_x, root_y, root_z, qw, qx, qy, qz, joint_0, ..., joint_28]."""

    save_gt_action_json_path: str | None = "./gt_action.json"
    """Path to save ground truth actions as JSON. Same format as save_action_json_path."""

def main(args: ArgsConfig):
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Download model checkpoint if it's an S3 path
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    global_step = None
    if local_model_path:
        # Search for pattern "checkpoint-{number}" anywhere in the path
        match = re.search(r"checkpoint-(\d+)", local_model_path)
        if match:
            try:
                global_step = int(match.group(1))
                logging.info(f"Extracted global_step {global_step} from checkpoint path")
            except ValueError:
                logging.warning(
                    f"Could not parse step number from checkpoint path: {local_model_path}"
                )
        else:
            logging.warning(f"Could not find checkpoint-<step> pattern in path: {local_model_path}")

    if local_model_path is not None:
        import torch

        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=local_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy = PolicyClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    logging.info(f"Current modality config: \n{modality}")

    # Create the dataset
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")

    all_mse = []
    all_mae = []

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        mse, mae = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            checkpoint_name=args.checkpoint_name,
            save_action_json_path=args.save_action_json_path,
            save_gt_action_json_path=args.save_gt_action_json_path,
        )
        logging.info(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
        all_mse.append(mse)
        all_mae.append(mae)

    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logging.info(f"Average MSE across all trajs: {avg_mse}")
        logging.info(f"Average MAE across all trajs: {avg_mae}")
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("Done")


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
