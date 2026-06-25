# Launch finetuning for N1.6 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def parse_dataset_path_groups(dataset_path_groups: list[str]) -> list[list[str]]:
    """Parse comma-separated dataset path groups from the finetune CLI."""
    parsed_groups = []
    for group in dataset_path_groups:
        paths = [path.strip() for path in group.split(",") if path.strip()]
        if len(paths) == 0:
            raise ValueError("--dataset-path-groups contains an empty dataset group")
        parsed_groups.append(paths)
    return parsed_groups


def get_mix_ratios(ft_config: FinetuneConfig, expected_len: int, flag_name: str) -> list[float]:
    if ft_config.dataset_mix_ratios is None:
        return [1.0] * expected_len

    if len(ft_config.dataset_mix_ratios) != expected_len:
        raise ValueError(
            "--dataset-mix-ratios must have the same length as "
            f"{flag_name} ({len(ft_config.dataset_mix_ratios)} != {expected_len})"
        )
    return ft_config.dataset_mix_ratios


def get_embodiment_tags(ft_config: FinetuneConfig, expected_len: int, flag_name: str) -> list[str]:
    if ft_config.dataset_embodiment_tags is None:
        if ft_config.embodiment_tag is None:
            raise ValueError("Either --embodiment-tag or --dataset-embodiment-tags must be provided")
        return [ft_config.embodiment_tag.value] * expected_len

    if len(ft_config.dataset_embodiment_tags) != expected_len:
        raise ValueError(
            "--dataset-embodiment-tags must have the same length as "
            f"{flag_name} ({len(ft_config.dataset_embodiment_tags)} != {expected_len})"
        )
    return [tag.value for tag in ft_config.dataset_embodiment_tags]


def build_dataset_configs(ft_config: FinetuneConfig) -> list[dict]:
    """Build data.datasets config for single-dataset or multi-dataset finetuning."""
    if ft_config.dataset_path_groups is not None:
        if len(ft_config.dataset_path_groups) == 0:
            raise ValueError("--dataset-path-groups must contain at least one dataset group")

        dataset_path_groups = parse_dataset_path_groups(ft_config.dataset_path_groups)
        mix_ratios = get_mix_ratios(
            ft_config, len(dataset_path_groups), "--dataset-path-groups"
        )
        embodiment_tags = get_embodiment_tags(
            ft_config, len(dataset_path_groups), "--dataset-path-groups"
        )

        return [
            {
                "dataset_paths": dataset_paths,
                "mix_ratio": mix_ratio,
                "embodiment_tag": dataset_embodiment_tag,
            }
            for dataset_paths, mix_ratio, dataset_embodiment_tag in zip(
                dataset_path_groups, mix_ratios, embodiment_tags
            )
        ]

    if ft_config.dataset_paths is not None:
        if len(ft_config.dataset_paths) == 0:
            raise ValueError("--dataset-paths must contain at least one dataset path")

        mix_ratios = get_mix_ratios(ft_config, len(ft_config.dataset_paths), "--dataset-paths")
        embodiment_tags = get_embodiment_tags(
            ft_config, len(ft_config.dataset_paths), "--dataset-paths"
        )

        return [
            {
                "dataset_paths": [dataset_path],
                "mix_ratio": mix_ratio,
                "embodiment_tag": dataset_embodiment_tag,
            }
            for dataset_path, mix_ratio, dataset_embodiment_tag in zip(
                ft_config.dataset_paths, mix_ratios, embodiment_tags
            )
        ]

    if ft_config.dataset_path is None:
        raise ValueError(
            "Either --dataset-path, --dataset-paths, or --dataset-path-groups must be provided"
        )

    if ft_config.dataset_mix_ratios is not None:
        raise ValueError(
            "--dataset-mix-ratios can only be used with "
            "--dataset-paths or --dataset-path-groups"
        )

    if ft_config.dataset_embodiment_tags is not None:
        raise ValueError(
            "--dataset-embodiment-tags can only be used with "
            "--dataset-paths or --dataset-path-groups"
        )

    if ft_config.embodiment_tag is None:
        raise ValueError("--embodiment-tag must be provided when using --dataset-path")

    return [
        {
            "dataset_paths": [ft_config.dataset_path],
            "mix_ratio": 1.0,
            "embodiment_tag": ft_config.embodiment_tag.value,
        }
    ]


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": build_dataset_configs(ft_config),
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.init_model_from = ft_config.init_model_from
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d6"

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    run(config)
