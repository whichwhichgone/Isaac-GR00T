"""
Unified processor for robot state and action data.

Handles:
- State normalization (min/max, mean/std, sin/cos encoding)
- Action normalization
- Absolute <-> Relative action representation conversion
- Action processing with state dependency
"""

from copy import deepcopy

from gr00t.configs.data.embodiment_configs import (
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.data.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.state_action.pose import EndEffectorPose, JointPose
from gr00t.data.utils import (
    apply_sin_cos_encoding,
    nested_dict_to_numpy,
    normalize_values_meanstd,
    normalize_values_minmax,
    parse_modality_configs,
    unnormalize_values_meanstd,
    unnormalize_values_minmax,
)
import numpy as np


_UNITREE_ACTION_LAYOUTS = {
    EmbodimentTag.UNITREE_G1_29DOF.value: {
        "action_dim": 102,
        "raw_slice": slice(36, 102),
    },
    EmbodimentTag.UNITREE_G1_29DOF_HAND.value: {
        "action_dim": 114,
        "raw_slice": slice(36, 102),
    },
    EmbodimentTag.UNITREE_G1_29DOF_HAND_SINGLE_VIEW.value: {
        "action_dim": 114,
        "raw_slice": slice(36, 102),
    },
    EmbodimentTag.UNITREE_G1_29DOF_HAND_NO_HISTORY.value: {
        "action_dim": 114,
        "raw_slice": slice(36, 102),
    },
}


class StateActionProcessor:
    """
    Unified processor for robot state and action data.

    Handles:
    - State normalization (min/max, mean/std, sin/cos encoding)
    - Action normalization
    - Absolute <-> Relative action representation conversion
    - Action processing with state dependency
    """

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: (dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None) = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = False,
    ):
        """
        Initialize unified state and action processor.

        Args:
            modality_configs: Nested dict with structure:
                {embodiment_tag: {modality: ModalityConfig}}
                where modality in ["state", "action"]
                Example: {"gr1": {"state": ModalityConfig(...), "action": ModalityConfig(...)}}
            statistics: Optional nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
                where modality in ["state", "action", "relative_action"]
                and stat_type in ["min", "max", "mean", "std", "q01", "q99"]
                Example: {"gr1": {"state": {"left_arm": {"min": [...], "max": [...], ...}}}}
            use_percentiles: Whether to use percentiles (q01/q99) instead of min/max
            clip_outliers: Whether to clip normalized values to [-1, 1]
            apply_sincos_state_encoding: Global flag to enable sin/cos encoding for states
        """
        self.modality_configs = parse_modality_configs(modality_configs)
        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action

        # Normalization parameters computed from statistics
        self.norm_params: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
        # Format: norm_params[embodiment_tag][modality][joint_group][stat_type]
        # where stat_type in ["min", "max", "mean", "std", "dim"]

        if statistics is not None:
            self.set_statistics(statistics)

        self.train()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def remap_rtc_normalized_action(
        self,
        normalized_action: np.ndarray,
        embodiment_tag: str,
        source_start: int,
        target_start: int,
        steps: int,
    ) -> np.ndarray:
        """Move an RTC action segment between time-indexed normalization slots.

        Unitree action statistics are computed over a flattened action chunk.
        Consequently, copying normalized action ``source_start`` to
        ``target_start`` directly changes its physical value.  This method
        decodes with the source slot statistics and re-encodes with the target
        slot statistics before the model freezes the prefix.

        Returns a ``(B, steps, D_model)`` segment normalized for target slots.
        """
        values = np.asarray(normalized_action)
        if values.ndim != 3:
            raise ValueError(
                "normalized_action must have shape (B, H, D), "
                f"got {values.shape}"
            )
        if source_start < 0 or target_start < 0 or steps < 0:
            raise ValueError(
                "RTC time indices must be non-negative, "
                f"got source={source_start}, target={target_start}, steps={steps}"
            )
        if source_start + steps > values.shape[1]:
            raise ValueError(
                "RTC source segment exceeds the model horizon: "
                f"source={source_start}, steps={steps}, H={values.shape[1]}"
            )

        segment = values[:, source_start : source_start + steps].copy()
        if steps == 0:
            return segment

        layout = _UNITREE_ACTION_LAYOUTS.get(embodiment_tag)
        if layout is None:
            # Other embodiments normally use time-shared per-channel stats.
            return segment

        action_dim = int(layout["action_dim"])
        if values.shape[2] < action_dim:
            raise ValueError(
                f"Model action dim {values.shape[2]} is smaller than physical dim {action_dim}"
            )

        action_config = self.modality_configs[embodiment_tag]["action"]
        if action_config.modality_keys != ["mocap"]:
            raise ValueError(
                "Unitree RTC normalization remapping expects one 'mocap' action group, "
                f"got {action_config.modality_keys}"
            )
        params = self.norm_params[embodiment_tag]["action"]["mocap"]
        stats_size = int(np.asarray(params["min"]).size)
        if stats_size % action_dim != 0:
            raise ValueError(
                f"Action statistics size {stats_size} is not divisible by action_dim={action_dim}"
            )
        stats_horizon = stats_size // action_dim
        if source_start + steps > stats_horizon or target_start + steps > stats_horizon:
            raise ValueError(
                "RTC segment exceeds the action-statistics horizon: "
                f"source={source_start}, target={target_start}, steps={steps}, "
                f"H_stats={stats_horizon}"
            )

        source_values = segment[..., :action_dim]
        use_meanstd = (
            action_config.mean_std_embedding_keys is not None
            and "mocap" in action_config.mean_std_embedding_keys
        )
        if use_meanstd:
            mean = np.asarray(params["mean"]).reshape(stats_horizon, action_dim)
            std = np.asarray(params["std"]).reshape(stats_horizon, action_dim)
            source_mean = mean[source_start : source_start + steps]
            source_std = std[source_start : source_start + steps]
            target_mean = mean[target_start : target_start + steps]
            target_std = std[target_start : target_start + steps]

            physical = np.where(
                source_std != 0,
                source_values * source_std + source_mean,
                source_values,
            )
            safe_target_std = np.where(target_std != 0, target_std, 1.0)
            remapped = (physical - target_mean) / safe_target_std
            remapped = np.where(target_std != 0, remapped, physical)
        else:
            min_values = np.asarray(params["min"]).reshape(stats_horizon, action_dim)
            max_values = np.asarray(params["max"]).reshape(stats_horizon, action_dim)
            source_min = min_values[source_start : source_start + steps]
            source_max = max_values[source_start : source_start + steps]
            target_min = min_values[target_start : target_start + steps]
            target_max = max_values[target_start : target_start + steps]

            # Match unnormalize_values_minmax: the previously executed physical
            # action was decoded after clipping the normalized value to [-1, 1].
            physical = (
                (np.clip(source_values, -1.0, 1.0) + 1.0)
                / 2.0
                * (source_max - source_min)
                + source_min
            )
            target_range = target_max - target_min
            safe_target_range = np.where(
                np.isclose(target_range, 0.0), 1.0, target_range
            )
            remapped = 2.0 * (physical - target_min) / safe_target_range - 1.0
            remapped = np.where(np.isclose(target_range, 0.0), 0.0, remapped)

        # Rotation-6D values deliberately bypass normalization in apply_action
        # and unapply_action, so they must also bypass this time-slot remapping.
        raw_slice = layout["raw_slice"]
        remapped[..., raw_slice] = source_values[..., raw_slice]
        segment[..., :action_dim] = remapped.astype(segment.dtype, copy=False)
        return segment

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """
        Set dataset statistics for normalization.

        Args:
            statistics: Nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
        """
        for key in statistics:
            if key not in self.statistics or override:
                self.statistics[key] = deepcopy(statistics[key])
            else:
                print(f"Embodiment tag {key} already in statistics, skipping updating")
        self._compute_normalization_parameters()

    def _compute_normalization_parameters(self) -> None:
        """Compute and cache normalization parameters from statistics for all embodiments and modalities."""
        for embodiment_tag in self.statistics:
            self.norm_params[embodiment_tag] = {}

            for modality in ["state", "action"]:
                if modality not in self.statistics[embodiment_tag]:
                    continue

                self.norm_params[embodiment_tag][modality] = {}

                for joint_group, stats in self.statistics[embodiment_tag][modality].items():
                    if self.use_percentiles:
                        min_vals = np.array(stats["q01"])
                        max_vals = np.array(stats["q99"])
                    else:
                        min_vals = np.array(stats["min"])
                        max_vals = np.array(stats["max"])

                    mean_vals = np.array(stats["mean"])
                    std_vals = np.array(stats["std"])

                    # Compute range, ensuring it's not zero
                    range_vals = max_vals - min_vals
                    range_vals = np.maximum(range_vals, 1e-8)

                    self.norm_params[embodiment_tag][modality][joint_group] = {
                        "min": min_vals,
                        "max": max_vals,
                        "dim": np.array(range_vals.shape[0]),
                        "mean": mean_vals,
                        "std": std_vals,
                    }

            # Override absolute action stats with relative stats where specified
            if "action" in self.modality_configs[embodiment_tag]:
                modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
                action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

                if action_configs is not None:
                    for key, action_config in zip(modality_keys, action_configs):
                        if (
                            action_config.rep == ActionRepresentation.RELATIVE
                            and self.use_relative_action
                        ):
                            if "relative_action" not in self.statistics[embodiment_tag]:
                                raise ValueError(
                                    f"Relative action statistics required for embodiment '{embodiment_tag}' "
                                    f"but 'relative_action' not found in statistics"
                                )
                            if key not in self.statistics[embodiment_tag]["relative_action"]:
                                raise ValueError(
                                    f"Relative action statistics required for key '{key}' "
                                    f"in embodiment '{embodiment_tag}' but not found"
                                )
                            action_dim = self.norm_params[embodiment_tag]["action"][key]["dim"]
                            self.norm_params[embodiment_tag]["action"][key] = nested_dict_to_numpy(
                                self.statistics[embodiment_tag]["relative_action"][key]
                            )
                            self.norm_params[embodiment_tag]["action"][key]["dim"] = action_dim

    def apply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """
        Apply state processing (normalization, encoding).

        Args:
            state: Dict mapping joint_group -> raw state values
                Shape per group: (..., D) where D is state dimension
            embodiment_tag: Embodiment identifier (e.g., "gr1")

        Returns:
            Dict mapping joint_group -> processed state values
                - Sin/cos encoded groups: (..., 2*D)
                - Other groups: (..., D)
        """
        unnormalized_imu = None
        if embodiment_tag in (
            EmbodimentTag.UNITREE_G1_29DOF.value,
        ):
            unnormalized_imu = state["imu_joints"].reshape(-1, 35)[:, :6].copy()
        if embodiment_tag in (
            EmbodimentTag.UNITREE_G1_29DOF_HAND.value, EmbodimentTag.UNITREE_G1_29DOF_HAND_SINGLE_VIEW.value, EmbodimentTag.UNITREE_G1_29DOF_HAND_NO_HISTORY.value
        ):
            unnormalized_imu = state["imu_joints"].reshape(-1, 47)[:, :6].copy()
        normalized_values = {}
        state = deepcopy(state)  # Avoid modifying input

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Strategy 1: Sin/cos encoding (doubles dimension)
            if sin_cos_keys and joint_group in sin_cos_keys:
                normalized_values[joint_group] = apply_sin_cos_encoding(state[joint_group])

            # Strategy 2: Mean/std normalization
            elif (
                hasattr(
                    self.modality_configs[embodiment_tag]["state"],
                    "mean_std_embedding_keys",
                )
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group
                in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_meanstd(state[joint_group], params)
                normalized_values[joint_group] = normalized

            # Strategy 3: Min/max normalization to [-1, 1]
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_minmax(state[joint_group], params)

                if self.clip_outliers:
                    normalized = np.clip(normalized, -1.0, 1.0)

                normalized_values[joint_group] = normalized

            if embodiment_tag == EmbodimentTag.UNITREE_G1_29DOF.value:
                imu_joints_shape = normalized_values["imu_joints"].shape
                imu_joints = normalized_values["imu_joints"].reshape(-1, 35)
                imu_joints[:, :6] = unnormalized_imu
                normalized_values["imu_joints"] = imu_joints.reshape(imu_joints_shape)
            if embodiment_tag in (EmbodimentTag.UNITREE_G1_29DOF_HAND.value, EmbodimentTag.UNITREE_G1_29DOF_HAND_SINGLE_VIEW.value, EmbodimentTag.UNITREE_G1_29DOF_HAND_NO_HISTORY.value):
                imu_joints_shape = normalized_values["imu_joints"].shape
                imu_joints = normalized_values["imu_joints"].reshape(-1, 47)
                imu_joints[:, :6] = unnormalized_imu
                normalized_values["imu_joints"] = imu_joints.reshape(imu_joints_shape)

        return normalized_values

    def unapply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """
        Reverse state processing (denormalization).

        Args:
            state: Dict mapping joint_group -> processed state values
            embodiment_tag: Embodiment identifier

        Returns:
            Dict mapping joint_group -> raw state values

        Raises:
            ValueError: If attempting to reverse sin/cos encoding (not reversible)
        """
        unnormalized_values = {}

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Sin/cos encoding is not reversible
            if sin_cos_keys and joint_group in sin_cos_keys:
                raise ValueError(
                    f"Cannot unapply sin/cos encoding for joint group '{joint_group}' "
                    f"in embodiment '{embodiment_tag}'. This transformation is not reversible."
                )

            # Reverse mean/std normalization
            elif (
                hasattr(
                    self.modality_configs[embodiment_tag]["state"],
                    "mean_std_embedding_keys",
                )
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group
                in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                unnormalized = unnormalize_values_meanstd(state[joint_group], params)
                unnormalized_values[joint_group] = unnormalized

            # Reverse min/max normalization
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                unnormalized_values[joint_group] = unnormalize_values_minmax(
                    state[joint_group], params
                )

        return unnormalized_values

    def apply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Apply action processing (absolute->relative conversion, normalization).

        Processing order:
        1. Convert absolute actions to relative (if configured)
        2. Normalize actions

        Args:
            action: Dict mapping joint_group -> raw action values
                Shape per group: (T, D) where T is action horizon, D is action dimension
            embodiment_tag: Embodiment identifier
            state: Optional dict mapping joint_group -> raw state values
                Required if any action group uses ActionRepresentation.RELATIVE
                Shape per group: (T_state, D) where last timestep is used as reference

        Returns:
            Dict mapping joint_group -> processed action values
                Shape per group: (T, D)

        Raises:
            ValueError: If state is None but required for relative action conversion
        """
        unnormalized_xyz_tail = None
        if embodiment_tag in (
            EmbodimentTag.UNITREE_G1_29DOF.value,
        ):
            unnormalized_xyz_tail = action["mocap"].reshape(-1, 102)[:, 36:].copy()
        if embodiment_tag in (
            EmbodimentTag.UNITREE_G1_29DOF_HAND.value, EmbodimentTag.UNITREE_G1_29DOF_HAND_SINGLE_VIEW.value, EmbodimentTag.UNITREE_G1_29DOF_HAND_NO_HISTORY.value
        ):
            unnormalized_xyz_tail = action["mocap"].reshape(-1, 114)[:, 36: 102].copy()
        action = deepcopy(action)  # Avoid modifying input

        # Step 1: Convert absolute actions to relative (if needed)
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

        if action_configs is not None:
            for key, action_config in zip(modality_keys, action_configs):
                if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                    if state is None:
                        raise ValueError(
                            f"State dict required for relative action processing of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )

                    # Determine which state key to use as reference
                    state_key = action_config.state_key if action_config.state_key else key

                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )

                    # Use last state as reference frame
                    reference_state = state[state_key][-1]

                    # Convert absolute to relative
                    action[key] = self._convert_to_relative_action(
                        action=action[key],
                        reference_state=reference_state,
                        action_type=action_config.type,
                        action_format=action_config.format,
                    )

        # Step 2: Normalize actions
        normalized_values = {}
        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            if (
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group
                in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                normalized = normalize_values_meanstd(action[joint_group], params)
            else:
                normalized = normalize_values_minmax(action[joint_group], params)

            if self.clip_outliers:
                normalized = np.clip(normalized, -1.0, 1.0)

            normalized_values[joint_group] = normalized

        if embodiment_tag == EmbodimentTag.UNITREE_G1_29DOF.value:
            mocap_shape = normalized_values["mocap"].shape
            mocap = normalized_values["mocap"].reshape(-1, 102)
            mocap[:, 36:] = unnormalized_xyz_tail
            normalized_values["mocap"] = mocap.reshape(mocap_shape)
        if embodiment_tag in (EmbodimentTag.UNITREE_G1_29DOF_HAND.value, EmbodimentTag.UNITREE_G1_29DOF_HAND_SINGLE_VIEW.value, EmbodimentTag.UNITREE_G1_29DOF_HAND_NO_HISTORY.value):
            mocap_shape = normalized_values["mocap"].shape
            mocap = normalized_values["mocap"].reshape(-1, 114)
            mocap[:, 36:102] = unnormalized_xyz_tail
            normalized_values["mocap"] = mocap.reshape(mocap_shape)
        return normalized_values

    def unapply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
        action_horizon: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Reverse action processing (denormalization, relative->absolute conversion).

        Processing order:
        1. Denormalize actions
        2. Convert relative actions to absolute (if configured)

        Args:
            action: Dict mapping joint_group -> processed action values
                Shape per group: (T, D) or (B, T, D) for batched
            embodiment_tag: Embodiment identifier
            state: Optional dict mapping joint_group -> raw state values
                Required if any action group uses ActionRepresentation.RELATIVE
                Shape per group: (T_state, D) or (B, T_state, D) for batched
            action_horizon: Number of physical steps returned to the caller. For
                flattened Unitree chunks this is independent of the statistics
                chunk size.

        Returns:
            Dict mapping joint_group -> raw absolute action values
                Shape per group: (T, D) or (B, T, D) for batched

        Raises:
            ValueError: If state is None but required for relative->absolute conversion
        """
        layout = _UNITREE_ACTION_LAYOUTS.get(embodiment_tag)
        if action_horizon is None:
            if layout is None:
                action_horizon = len(
                    self.modality_configs[embodiment_tag]["action"].delta_indices
                )
            else:
                mocap_values = np.asarray(action["mocap"])
                values_per_batch = (
                    int(np.prod(mocap_values.shape[1:]))
                    if mocap_values.ndim == 3
                    else mocap_values.size
                )
                action_dim = int(layout["action_dim"])
                if values_per_batch % action_dim != 0:
                    raise ValueError(
                        f"Flattened action size {values_per_batch} is not divisible "
                        f"by action_dim={action_dim}"
                    )
                action_horizon = values_per_batch // action_dim
        action_horizon = int(action_horizon)
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        # Step 1: Unnormalize actions
        unnormalized_values = {}
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys

        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            group_values = action[joint_group]

            if (
                self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys is not None
                and joint_group
                in self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys
            ):
                unnormalized = unnormalize_values_meanstd(group_values, params)
            else:
                unnormalized = unnormalize_values_minmax(group_values, params)

            unnormalized_values[joint_group] = unnormalized

        if embodiment_tag == EmbodimentTag.UNITREE_G1_29DOF.value:
            mocap_values = unnormalized_values["mocap"]
            raw_values = np.asarray(action["mocap"])
            if mocap_values.ndim == 3:
                batch_size = mocap_values.shape[0]
                mocap = mocap_values.reshape(batch_size, -1, 102)
                raw_mocap = raw_values.reshape(batch_size, -1, 102)
                if action_horizon > mocap.shape[1]:
                    raise ValueError(
                        f"action_horizon={action_horizon} exceeds decoded chunk "
                        f"size {mocap.shape[1]}"
                    )
                mocap[..., 36:] = raw_mocap[..., 36:]
                unnormalized_values["mocap"] = mocap[:, :action_horizon].reshape(
                    batch_size, 1, -1
                )
            else:
                mocap = mocap_values.reshape(-1, 102)
                raw_mocap = raw_values.reshape(-1, 102)
                if action_horizon > mocap.shape[0]:
                    raise ValueError(
                        f"action_horizon={action_horizon} exceeds decoded chunk "
                        f"size {mocap.shape[0]}"
                    )
                mocap[:, 36:] = raw_mocap[:, 36:]
                unnormalized_values["mocap"] = mocap[:action_horizon].reshape(-1)
        if embodiment_tag in (
            EmbodimentTag.UNITREE_G1_29DOF_HAND.value,
            EmbodimentTag.UNITREE_G1_29DOF_HAND_SINGLE_VIEW.value,
            EmbodimentTag.UNITREE_G1_29DOF_HAND_NO_HISTORY.value,
        ):
            mocap_values = unnormalized_values["mocap"]
            raw_values = np.asarray(action["mocap"])
            if mocap_values.ndim == 3:
                batch_size = mocap_values.shape[0]
                mocap = mocap_values.reshape(batch_size, -1, 114)
                raw_mocap = raw_values.reshape(batch_size, -1, 114)
                if action_horizon > mocap.shape[1]:
                    raise ValueError(
                        f"action_horizon={action_horizon} exceeds decoded chunk "
                        f"size {mocap.shape[1]}"
                    )
                mocap[..., 36:102] = raw_mocap[..., 36:102]
                # The public modality has T=1 and stores the executable chunk in
                # its feature axis: (B, 1, action_horizon * 114). Preserve batch and
                # modality-time dimensions for strict policy validation.
                unnormalized_values["mocap"] = mocap[:, :action_horizon].reshape(
                    batch_size, 1, -1
                )
            else:
                # Retain the unbatched convention for direct processor callers.
                mocap = mocap_values.reshape(-1, 114)
                raw_mocap = raw_values.reshape(-1, 114)
                if action_horizon > mocap.shape[0]:
                    raise ValueError(
                        f"action_horizon={action_horizon} exceeds decoded chunk "
                        f"size {mocap.shape[0]}"
                    )
                mocap[:, 36:102] = raw_mocap[:, 36:102]
                unnormalized_values["mocap"] = mocap[:action_horizon].reshape(-1)

        # Step 2: Convert relative actions to absolute (if needed)
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

        if action_configs is not None:
            for key, action_config in zip(modality_keys, action_configs):
                if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                    if state is None:
                        raise ValueError(
                            f"State dict required for relative->absolute conversion of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )

                    # Determine which state key to use as reference
                    state_key = action_config.state_key if action_config.state_key else key

                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )

                    relative_action = unnormalized_values[key]

                    # Handle batched and unbatched cases
                    is_batched = relative_action.ndim == 3
                    if not is_batched:
                        assert relative_action.ndim == 2
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            reference_state = reference_state[None, :]
                        relative_action = relative_action[None, :]
                    else:
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            reference_state = reference_state[None, :]

                    # Convert batched relative actions to absolute
                    absolute_actions = []
                    for s, a in zip(reference_state, relative_action):
                        # Use last timestep of state as reference
                        absolute_action = self._convert_to_absolute_action(
                            action=a,
                            reference_state=s[-1],
                            action_type=action_config.type,
                            action_format=action_config.format,
                        )
                        absolute_actions.append(absolute_action)

                    if is_batched:
                        unnormalized_values[key] = np.stack(absolute_actions, axis=0)
                    else:
                        unnormalized_values[key] = absolute_actions[0]

        return unnormalized_values

    def apply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Apply both state and action processing together.

        Convenience method that processes state and action in one call,
        automatically passing raw state to action processor for relative conversion.

        Args:
            state: Dict mapping joint_group -> raw state values
            action: Dict mapping joint_group -> raw action values
            embodiment_tag: Embodiment identifier

        Returns:
            Tuple of (processed_state, processed_action)
        """
        processed_state = self.apply_state(state, embodiment_tag)
        if action:
            processed_action = self.apply_action(action, embodiment_tag, state=state)
        else:
            assert not self.training, "Action is required in training mode"
            processed_action = {}
        return processed_state, processed_action

    def unapply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        raw_state: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Reverse both state and action processing together.

        Args:
            state: Dict mapping joint_group -> processed state values
            action: Dict mapping joint_group -> processed action values
            embodiment_tag: Embodiment identifier
            raw_state: Optional dict of raw states for relative->absolute conversion
                If None, will use unapplied state (but won't work for sin/cos encoded states)

        Returns:
            Tuple of (raw_state, raw_action)
        """
        # Unapply state first
        try:
            unapplied_state = self.unapply_state(state, embodiment_tag)
        except ValueError as e:
            if "sin/cos encoding" in str(e) and raw_state is None:
                raise ValueError(
                    "Cannot unapply sin/cos encoded state. Please provide raw_state parameter."
                ) from e
            raise

        # Use provided raw_state if available, otherwise use unapplied state
        state_for_action = raw_state if raw_state is not None else unapplied_state

        # Unapply action
        unapplied_action = self.unapply_action(action, embodiment_tag, state=state_for_action)

        return unapplied_state, unapplied_action

    def get_state_dim(self, embodiment_tag: str, include_sincos_expansion: bool = False) -> int:
        """
        Get total state dimension after processing.

        Args:
            embodiment_tag: Embodiment identifier
            include_sincos_expansion: If True, accounts for sin/cos encoding doubling dimensions

        Returns:
            Total state dimension across all joint groups
        """
        total_dim = 0
        state_config = self.modality_configs[embodiment_tag]["state"]

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = set()
        if self.apply_sincos_state_encoding and hasattr(state_config, "sin_cos_embedding_keys"):
            sin_cos_keys = set(state_config.sin_cos_embedding_keys)

        for joint_group in state_config.modality_keys:
            base_dim = self.norm_params[embodiment_tag]["state"][joint_group]["dim"].item()

            # Sin/cos encoding doubles the dimension
            if include_sincos_expansion and joint_group in sin_cos_keys:
                total_dim += base_dim * 2
            else:
                total_dim += base_dim

        return total_dim

    def get_action_dim(self, embodiment_tag: str) -> int:
        """
        Get total action dimension.

        Args:
            embodiment_tag: Embodiment identifier

        Returns:
            Total action dimension across all joint groups
        """
        total_dim = 0
        for joint_group in self.modality_configs[embodiment_tag]["action"].modality_keys:
            total_dim += self.norm_params[embodiment_tag]["action"][joint_group]["dim"].item()
        return total_dim

    def _convert_to_relative_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        """Convert absolute action to relative action using reference state."""
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"

        if action_type == ActionType.EEF:
            action_chunking = EndEffectorActionChunk.from_array(action, action_format)
            reference_frame = EndEffectorPose.from_action_format(reference_state, action_format)

        elif action_type == ActionType.NON_EEF:
            action_chunking = JointActionChunk([JointPose(m) for m in action])
            reference_frame = JointPose(reference_state)

        else:
            raise ValueError(f"Unknown ActionType: {action_type}")

        relative_action_chunking = action_chunking.relative_chunking(
            reference_frame=reference_frame
        )
        return relative_action_chunking.to(action_format)

    def _convert_to_absolute_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        """Convert relative action to absolute action using reference state."""
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"
        assert reference_state.shape[0] == action.shape[1], (
            f"State dim {reference_state.shape[0]} != action dim {action.shape[1]}"
        )

        if action_type == ActionType.EEF:
            rel_action = EndEffectorActionChunk.from_array(action, action_format)
            reference_frame = EndEffectorPose.from_action_format(reference_state, action_format)

        elif action_type == ActionType.NON_EEF:
            rel_action = JointActionChunk([JointPose(pose) for pose in action])
            reference_frame = JointPose(reference_state)

        else:
            raise ValueError(f"Unknown ActionType: {action_type}")

        abs_action = rel_action.to_absolute_chunking(reference_frame=reference_frame)
        return abs_action.to(action_format)

    def __str__(self) -> str:
        return f"StateActionProcessor(modality_configs={self.modality_configs}, statistics={self.statistics}, use_percentiles={self.use_percentiles}, clip_outliers={self.clip_outliers}, apply_sincos_state_encoding={self.apply_sincos_state_encoding}, use_relative_action={self.use_relative_action})"
