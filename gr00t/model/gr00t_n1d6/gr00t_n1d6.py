from typing import Tuple

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree
import math as th
from typing import Optional


class Gr00tN1d6ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d6Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Initialize components directly from config
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            print("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg, cross_attention_dim=config.backbone_embedding_dim
            )
            print("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob
        self.mask_token = (
            nn.Parameter(0.02 * torch.randn(1, 1, self.input_embedding_dim))
            if self.state_dropout_prob > 0
            else None
        )

        # State noise parameters
        self.state_additive_noise_scale = config.state_additive_noise_scale

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.state_dropout_prob > 0:
                self.mask_token.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        print(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features.
        if self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

        # Add Gaussian noise to state features.
        if self.training and self.state_additive_noise_scale > 0:
            print(
                f"Adding Gaussian noise to state features with scale {self.state_additive_noise_scale}"
            )
            noise = torch.randn_like(state_features) * self.state_additive_noise_scale
            state_features = state_features + noise

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, state_horizon, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @staticmethod
    def _rtc_softmask(
        H: int,
        d: int,
        s: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build RTC soft mask W.

        Args:
            H: action_horizon
            d: inference delay in action steps
            s: executed steps since last inference started

        Returns:
            W: [1, H, 1]
        """
        if d < 0:
            raise ValueError(f"d must be >= 0, got {d}")
        if s < 0:
            raise ValueError(f"s must be >= 0, got {s}")

        # RTC feasible condition from the paper:
        # d <= s <= H - d
        if d > H - d:
            raise ValueError(
                f"RTC infeasible: d={d}, H={H}. Need d <= H - d. "
                f"Increase action_horizon or reduce inference delay."
            )
        if not (d <= s <= H - d):
            raise ValueError(f"RTC requires d <= s <= H - d, got d={d}, s={s}, H={H}.")
        i = torch.arange(H, device=device, dtype=dtype)
        W = torch.zeros(H, device=device, dtype=dtype)

        # 1) frozen region: i < d
        W[:d] = 1.0

        # 2) intermediate region: d <= i < H - s
        overlap_end = H - s
        if overlap_end > d:
            c = (overlap_end - i[d:overlap_end]) / (overlap_end - d + 1)
            W[d:overlap_end] = c * (torch.exp(c) - 1) / (th.e - 1)

        return W.view(1, H, 1)

    @staticmethod
    def _rtc_align_previous_actions(
        previous_actions: torch.Tensor,
        executed_steps: int,
    ) -> torch.Tensor:
        """Align an old action chunk to the new inference start time."""
        if previous_actions.ndim != 3:
            raise ValueError(
                "previous_actions must have shape (B, H, D), "
                f"got {tuple(previous_actions.shape)}"
            )

        H = previous_actions.shape[1]
        if not 0 <= executed_steps <= H:
            raise ValueError(
                f"executed_steps must be in [0, {H}], got {executed_steps}"
            )

        aligned_actions = torch.zeros_like(previous_actions)
        aligned_actions[:, : H - executed_steps] = previous_actions[:, executed_steps:]
        return aligned_actions

    @staticmethod
    def _rtc_guidance_scale(
        tau: float,
        beta: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute min(beta, (1 - tau) / (tau * r_tau^2)).
        At tau=0, use beta because the raw value is singular and then clipped.
        """
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if tau <= 0.0:
            return torch.tensor(beta, device=device, dtype=dtype)

        tau_t = torch.tensor(tau, device=device, dtype=dtype)
        one = torch.ones((), device=device, dtype=dtype)
        eps = torch.tensor(1e-6, device=device, dtype=dtype)

        r2 = ((one - tau_t) ** 2) / (tau_t**2 + (one - tau_t) ** 2 + eps)
        raw = (one - tau_t) / (tau_t * r2 + eps)
        return torch.clamp(raw, max=beta)

    def get_action_with_features_rtc(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        rtc_prev_actions: Optional[torch.Tensor] = None,
        rtc_executed_steps: Optional[int] = None,
        rtc_delay_steps: Optional[int] = None,
        rtc_beta: float = 1.0,
    ) -> BatchFeature:
        """
        Generate actions using flow matching with optional RTC guidance.

        Args:
            backbone_features:
                [B, seq_len, backbone_embedding_dim]

            state_features:
                [B, state_horizon, input_embedding_dim]

            embodiment_id:
                [B]

            backbone_output:
                Output from the backbone model.

            rtc_prev_actions:
                Previous full, unaligned action chunk, shape [B, H, action_dim].
                This should be the previous model-predicted chunk, not interpolated actions.
                If None, this function falls back to normal flow inference.

            rtc_executed_steps:
                s in the paper. Number of action steps already executed from rtc_prev_actions
                when this new inference starts.

            rtc_delay_steps:
                d in the paper. Estimated inference delay measured in action steps.

            rtc_beta:
                Maximum guidance weight beta.
        """
        # In RTC, we need gradients w.r.t. actions, but not w.r.t. backbone/state features.
        vl_embeds = backbone_features.detach()
        state_features = state_features.detach()

        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        dtype = vl_embeds.dtype
        H = self.config.action_horizon

        actions = torch.randn(
            size=(batch_size, H, self.action_dim),
            dtype=dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps

        rtc_inputs = (rtc_prev_actions, rtc_executed_steps, rtc_delay_steps)
        provided_rtc_inputs = sum(value is not None for value in rtc_inputs)
        if provided_rtc_inputs not in (0, len(rtc_inputs)):
            raise ValueError(
                "RTC requires rtc_prev_actions, rtc_executed_steps, and rtc_delay_steps "
                "to be provided together."
            )
        use_rtc = provided_rtc_inputs == len(rtc_inputs)

        if use_rtc:
            s = int(rtc_executed_steps)
            d = int(rtc_delay_steps)

            rtc_prev_actions = torch.as_tensor(
                rtc_prev_actions,
                device=device,
                dtype=dtype,
            ).detach()
            if rtc_prev_actions.shape != (batch_size, H, self.action_dim):
                raise ValueError(
                    f"rtc_prev_actions shape error: expected "
                    f"{(batch_size, H, self.action_dim)}, got {rtc_prev_actions.shape}"
                )

            # Align the old chunk to the new inference start time as defined by RTC:
            # A_prev[i] = A_old[s + i], right-padded with zeros outside the overlap.
            A_prev = self._rtc_align_previous_actions(rtc_prev_actions, s)

            W = self._rtc_softmask(
                H=H,
                d=d,
                s=s,
                device=device,
                dtype=dtype,
            )
        else:
            A_prev = None
            W = None

        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(batch_size,),
                fill_value=t_discretized,
                device=device,
                dtype=torch.long,
            )

            def forward_velocity(action_input: torch.Tensor) -> torch.Tensor:
                """
                Compute v_pi(action_input, observation, tau).
                Returns:
                    pred_velocity: [B, H, action_dim]
                """
                action_features = self.action_encoder(
                    action_input,
                    timesteps_tensor,
                    embodiment_id,
                )

                if self.config.add_pos_embed:
                    pos_ids = torch.arange(
                        action_features.shape[1],
                        dtype=torch.long,
                        device=device,
                    )
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                sa_embs = torch.cat((state_features, action_features), dim=1)

                if self.config.use_alternate_vl_dit:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                        image_mask=backbone_output.image_mask,
                        backbone_attention_mask=backbone_output.backbone_attention_mask,
                    )
                else:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                    )

                pred = self.action_decoder(model_output, embodiment_id)
                pred_velocity = pred[:, -H:, :]
                return pred_velocity

            if use_rtc:
                with torch.enable_grad():
                    # Need graph from actions -> pred_velocity -> A_hat_1.
                    actions = actions.detach().requires_grad_(True)

                    # Original flow velocity.
                    pred_velocity = forward_velocity(actions)

                    # Eq. 3:
                    # A_hat_1 = A_tau + (1 - tau) * v_pi(A_tau, o, tau)
                    A_hat_1 = actions + (1.0 - t_cont) * pred_velocity

                    # Eq. 2 weighted error:
                    # e = (A_prev - A_hat_1)^T diag(W)
                    grad_outputs = (A_prev - A_hat_1) * W

                    # Vector-Jacobian product:
                    # g = e * d A_hat_1 / d A_tau
                    g = torch.autograd.grad(
                        outputs=A_hat_1,
                        inputs=actions,
                        grad_outputs=grad_outputs,
                        retain_graph=False,
                        create_graph=False,
                        only_inputs=True,
                    )[0]

                guidance_scale = self._rtc_guidance_scale(
                    tau=t_cont,
                    beta=rtc_beta,
                    device=device,
                    dtype=dtype,
                )

                # Eq. 1 with RTC guidance:
                # A_{tau+dt} = A_tau + dt * (v_pi + scale * g)
                with torch.no_grad():
                    actions = actions + dt * (pred_velocity + guidance_scale * g)

            else:
                # Normal non-RTC flow inference.
                with torch.no_grad():
                    actions = actions.detach()
                    pred_velocity = forward_velocity(actions)
                    actions = actions + dt * pred_velocity

        return BatchFeature(
            data={
                "action_pred": actions.detach(),
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
        )

    def get_action_rtc(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        previous_actions_rel,
        action_executed_steps,
        delay_frames,
        beta=5.0,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        with torch.no_grad():
            features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features_rtc(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            rtc_prev_actions=previous_actions_rel,
            rtc_executed_steps=action_executed_steps,
            rtc_delay_steps=delay_frames,
            rtc_beta=beta,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d6Config):
    if "NVEagle" in config.model_name or "nvidia/Eagle" in config.model_name:
        return EagleBackbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d6(PreTrainedModel):
    """Gr00tN1d6: Vision-Language-Action model with backbone."""

    config_class = Gr00tN1d6Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d6Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d6 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d6ActionHead(config)
        from .processing_gr00t_n1d6 import Gr00tN1d6DataCollator

        self.collator = Gr00tN1d6DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Eagle inputs (prefixed with 'eagle_')
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        return action_outputs

    def get_action_rtc(
        self,
        inputs: dict,
        previous_actions_rel,
        action_executed_steps,
        delay_frames,
        beta=5.0,
    ) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        with torch.no_grad():
            backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action_rtc(
            backbone_output=backbone_outputs,
            action_input=action_inputs,
            previous_actions_rel=previous_actions_rel,
            action_executed_steps=action_executed_steps,
            delay_frames=delay_frames,
            beta=beta,
        )

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
