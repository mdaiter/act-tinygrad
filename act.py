from itertools import chain
from copy import deepcopy

from tinygrad import Tensor, nn, dtypes
from config import ACTConfig
from layers import *
from normalize import Normalize, Unnormalize
from utils import *
from resnet import ResNetInstances

class ACT:
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        self.use_robot_state = "observation.state" in config.input_shapes
        self.use_images = any(k.startswith("observation.image") for k in config.input_shapes)
        self.use_env_state = "observation.environment_state" in config.input_shapes
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.use_robot_state:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    config.input_shapes["observation.state"][0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                config.output_shapes["action"][0], config.dim_model
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.use_robot_state:
                num_input_token_encoder += 1
            self.vae_encoder_pos_enc = create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0)
            self.vae_encoder_pos_enc.requires_grad = False

        # Backbone for image feature extraction.
        if self.use_images:
            backbone_model = ResNetInstances.resnet18_IMAGENET1K_V1
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = backbone_model #IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.use_robot_state:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.input_shapes["observation.state"][0], config.dim_model
            )
        if self.use_env_state:
            self.encoder_env_state_input_proj = nn.Linear(
                config.input_shapes["observation.environment_state"][0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.use_images:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                512, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.use_robot_state:
            n_1d_tokens += 1
        if self.use_env_state:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.use_images:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, config.output_shapes["action"][0])

        self._reset_parameters()

        # CHANGE THIS WHEN RUNNING.
        self.training=True

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(nn.state.get_parameters(self.encoder), nn.state.get_parameters(self.decoder)):
            if p.ndim > 1:
                def xavier_uniform_(tensor: Tensor) -> Tensor:
                    fan_in, fan_out = tensor.shape[:2]
                    
                    # Calculate the range for the uniform distribution
                    # This is the glorot/xavier uniform initialization formula
                    a = math.sqrt(6.0 / (fan_in + fan_out))
                    
                    # Use uniform distribution to initialize the tensor
                    return Tensor.uniform(*tensor.shape, low=-a, high=a)
                p = xavier_uniform_(p)

    def __call__(
            self, 
            observation_state: Tensor | None = None,
            observation_images: Tensor | None = None,
            observation_environment_state: Tensor | None = None,
            action: Tensor | None = None,
            action_is_pad: Tensor | None = None
        ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            "observation.state" (optional): (B, state_dim) batch of robot states.

            "observation.images": (B, n_cameras, C, H, W) batch of images.
                AND/OR
            "observation.environment_state": (B, env_dim) batch of environment states.

            "action" (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.config.use_vae and self.training:
            assert (
                action is not None
            ), "actions must be provided when using the variational objective in training mode."

        batch_size = (
            observation_images
            if observation_images is not None
            else observation_environment_state
        ).shape[0]

        print(f'batch_size: {batch_size}. Using observation_images? {observation_images is not None}')
        print(f'using vae? {self.config.use_vae and action is not None}')

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and action is not None:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = self.vae_encoder_cls_embed.weight.repeat(batch_size, 1, 1) # (B, 1, D)
            if self.use_robot_state:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(observation_state)
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(action)  # (B, S, D)

            if self.use_robot_state:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = Tensor.cat(*vae_encoder_input, dim=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.contiguous().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = Tensor.full(
                shape=(batch_size, 2 if self.use_robot_state else 1),
                fill_value=False
            )
            key_padding_mask = Tensor.cat(
                cls_joint_is_pad, action_is_pad, dim=1
            )  # (bs, seq+1 or 2)

            print(f'vae_encoder_input.shape: {vae_encoder_input.shape}')
            print(f'pos_embed.shape: {pos_embed.shape}')
            print(f'key_padding_mask.shape: {key_padding_mask.shape}')

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask.permute(1,0),
            )
            print(f'cls_token_out.shape: {cls_token_out.shape}')
            cls_token_out = cls_token_out[0]  # select the class token, with shape (B, D)
            print(f'cls_token_out[0].shape: {cls_token_out.shape}')
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * Tensor.randn(*(mu.shape))
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = Tensor.zeros(batch_size, self.config.latent_dim, dtype=dtypes.float32)

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.use_robot_state:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(observation_state))
        # Environment state token.
        if self.use_env_state:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(observation_environment_state)
            )

        # Camera observation features and positional embeddings.
        if self.use_images:
            all_cam_features = []
            all_cam_pos_embeds = []

            for cam_index in range(observation_images.shape[-4]):
                cam_features = self.backbone(observation_images[:, cam_index])  #["feature_map"]
                print(f'backbone output: {cam_features.shape}')
                # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use
                # buffer
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).cast(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, C, h, w)
                print(f'cam_features: {cam_features.shape}')
                all_cam_features.append(cam_features)
                print(f'len all_cam_features: {len(all_cam_features)}')
                all_cam_pos_embeds.append(cam_pos_embed)
            # Concatenate camera observation feature maps and positional embeddings along the width dimension,
            # and move to (sequence, batch, dim).
            all_cam_features = Tensor.cat(*all_cam_features, dim=-1)
            print(f'len all_cam_features after cat: {len(all_cam_features)}')
            print(f'Before encoder_in_tokens.extend, encoder_in token len: {len(encoder_in_tokens)}')
            encoder_in_tokens.extend(all_cam_features.permute(2, 3, 0, 1).reshape(-1, all_cam_features.shape[0], all_cam_features.shape[1]))
            print(f'encoder_in_tokens: {len(encoder_in_tokens)}')
            all_cam_pos_embeds = Tensor.cat(*all_cam_pos_embeds, dim=-1)
            print(f'all_cam_pos_embeds: {all_cam_pos_embeds}')
            encoder_in_pos_embed.extend(all_cam_pos_embeds.permute(2, 3, 0, 1).reshape(-1, all_cam_pos_embeds.shape[0], all_cam_pos_embeds.shape[1]))

        print(f'Before tensor.stack, encoder_in token len: {len(encoder_in_tokens)}')
        print(f'Before tensor.stack, encoder_in_pos_embed token len: {len(encoder_in_pos_embed)}')
        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = Tensor.stack(*encoder_in_tokens, dim=0)
        encoder_in_pos_embed = Tensor.stack(*encoder_in_pos_embed, dim=0)

        print(f'encoder_in_tokens: {len(encoder_in_tokens)}')
        print(f'encoder_in_pos_embed.shape: {encoder_in_pos_embed.shape}')

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = Tensor.zeros(
            *(self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype
        )
        print(f'encoder_out.shape: {encoder_out.shape}')
        print(f'decoder_in.shape: {decoder_in.shape}')
        print(f'encoder_in_pos_embed.shape: {encoder_in_pos_embed.shape}')
        print(f'decoder_pos_embed.shape: {self.decoder_pos_embed.weight.shape}')
        print(f'decoder_pos_embed.shape unsqueezed: {self.decoder_pos_embed.weight.unsqueeze(1).shape}')
        decoder_out = self.decoder(
            decoder_in.permute(1,0,2),
            encoder_out.permute(1,0,2),
            encoder_pos_embed=encoder_in_pos_embed.permute(1,0,2),
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1).permute(1,0,2),
        )

        # Move back to (B, S, C).
        # decoder_out = decoder_out.transpose(0, 1)
        print(f'decoder_out: {decoder_out.shape}')

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)

class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://arxiv.org/abs/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[:i+1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = (-temporal_ensemble_coeff * Tensor.arange(chunk_size)).exp()
        self.ensemble_weights_cumsum = self.ensemble_weights.cumsum(axis=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.contiguous()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = Tensor.ones(
                *(self.chunk_size, 1), dtype=dtypes.long
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = (self.ensembled_actions_count + 1).clamp(max_=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = Tensor.cat(*[self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = Tensor.cat(
                *[self.ensembled_actions_count, Tensor.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action

class ACTPolicy:
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    name = "act"

    def __init__(
        self,
        config: ACTConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = ACTConfig()
        self.config: ACTConfig = config

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self.model = ACT(config)

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        Tensor.no_grad = True

        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = Tensor.stack(*[batch[k] for k in self.expected_image_keys], dim=-4)

        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.model(batch)[0]  # (batch_size, chunk_size, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model(
                batch["observation.state"].realize(),
                batch["observation.images"].realize(),
                None,
                None,
                None
            )[0][:, : self.config.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        item_to_return = self._action_queue.popleft()
        Tensor.no_grad = False
        return item_to_return

    def normalize_batch_inputs_and_targets(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch) # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = Tensor.stack(*[batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch)
        return batch

    def __call__(
            self,
            observation_state: Tensor | None = None,
            observation_images: Tensor | None = None,
            observation_environment_state: Tensor | None = None,
            action: Tensor | None = None,
            action_is_pad: Tensor | None = None
        ) -> dict[str, Tensor]:
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(
            observation_state, observation_images, observation_environment_state, action, action_is_pad
        )

        l1_loss = (
            (action - actions_hat).abs() * action_is_pad.logical_not().int().unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://arxiv.org/abs/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.square() - (log_sigma_x2_hat).exp())).sum(axis=-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss_dict["loss"] = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss_dict["loss"] = l1_loss

        return loss_dict
