from tinygrad import Tensor, nn

from config import *
from utils import *
from networks import *

class ACTDecoderLayer:
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout_rate = config.dropout

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def __call__(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, x)  
        #x = x[0] # select just the output, not the attention weights
        x = skip + x.dropout(p=self.dropout_rate)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            self.maybe_add_pos_embed(x, decoder_pos_embed),
            self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            encoder_out,
        )
        #x = x[0]  # select just the output, not the attention weights
        x = skip + x.dropout(p=self.dropout_rate)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        
        x = x.sequential([self.linear1, self.activation]).dropout(p=self.dropout_rate).sequential([self.linear2])
        x = skip + x.dropout(p=self.dropout_rate)
        if not self.pre_norm:
            x = self.norm3(x)
        return x

class ACTDecoder:
    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = [ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)]
        self.norm = nn.LayerNorm(config.dim_model)

    def __call__(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x

class ACTEncoderLayer:
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = config.dropout
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def __call__(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, x, key_padding_mask=key_padding_mask)
        # x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + x.dropout(p=self.dropout)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = x.sequential([self.linear1, self.activation]).dropout(p=self.dropout).sequential([self.linear2])
        x = skip + x.dropout(p=self.dropout)
        if not self.pre_norm:
            x = self.norm2(x)
        return x

class ACTEncoder:
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = [ACTEncoderLayer(config) for _ in range(num_layers)]
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else lambda x: x

    def __call__(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            print(f'ACTEncoder x.shape per layer: {x.shape}')
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x

