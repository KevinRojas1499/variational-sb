import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from gluonts.torch.util import repeat_along_dim, unsqueeze_expand
from pts.util import lagged_sequence_values

class DiffusionEmbedding(nn.Module):
    def __init__(self, proj_dim):
        super().__init__()
        self.projection1 = nn.Linear(1, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.projection1(diffusion_step)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

class ContDiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = diffusion_step.unsqueeze(-1) * self.embedding
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim):
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        return 10.0 ** (dims * 4.0 / dim)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):
    def __init__(self, cond_dim, target_dim):
        super().__init__()
        self.linear1 = nn.Linear(cond_dim, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


    def prepare_rnn_input(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ):
        context = past_target[:, -self.context_length :, ...]
        observed_context = past_observed_values[:, -self.context_length :, ...]

        input, loc, scale = self.scaler(context, observed_context)
        future_length = future_time_feat.shape[-2]
        if future_length > 1:
            assert future_target is not None
            input = torch.cat(
                (input, (future_target[:, : future_length - 1, ...] - loc) / scale),
                dim=1,
            )
        prior_input = (past_target[:, : -self.context_length, ...] - loc) / scale

        lags = lagged_sequence_values(self.lags_seq, prior_input, input, dim=1)
        time_feat = torch.cat(
            (past_time_feat[:, -self.context_length + 1 :, ...], future_time_feat),
            dim=1,
        )

        embedded_cat = self.embedder(feat_static_cat)
        log_abs_loc = (
            loc.abs().log1p() if self.input_size == 1 else loc.squeeze(1).abs().log1p()
        )
        log_scale = scale.log() if self.input_size == 1 else scale.squeeze(1).log()

        static_feat = torch.cat(
            (embedded_cat, feat_static_real, log_abs_loc, log_scale), dim=-1
        )
        expanded_static_feat = unsqueeze_expand(
            static_feat, dim=1, size=time_feat.shape[-2]
        )

        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        return torch.cat((lags, features), dim=-1), loc, scale, static_feat

    def unroll_lagged_rnn(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ):
        """
        Applies the underlying RNN to the provided target data and covariates.

        Parameters
        ----------
        feat_static_cat
            Tensor of static categorical features,
            shape: ``(batch_size, num_feat_static_cat)``.
        feat_static_real
            Tensor of static real features,
            shape: ``(batch_size, num_feat_static_real)``.
        past_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, past_length, num_feat_dynamic_real)``.
        past_target
            Tensor of past target values,
            shape: ``(batch_size, past_length)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            Tensor of dynamic real features in the future,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        future_target
            (Optional) tensor of future target values,
            shape: ``(batch_size, prediction_length)``.

        Returns
        -------
        Tuple
            A tuple containing, in this order:
            - Parameters of the output distribution
            - Scaling factor applied to the target
            - Raw output of the RNN
            - Static input to the RNN
            - Output state from the RNN
        """
        rnn_input, loc, scale, static_feat = self.prepare_rnn_input(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )
        output, new_state = self.rnn(rnn_input)

        return loc, scale, output, static_feat, new_state


class EpsilonTheta(nn.Module):
    def __init__(
        self,
        target_dim,
        cond_dim,
        interval,
        time_emb_dim=16,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
    ):
        super().__init__()
        self.interval = interval
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.diffusion_embedding = DiffusionEmbedding(
            proj_dim=residual_hidden
        )
        self.cond_upsampler = CondUpsampler(target_dim=target_dim, cond_dim=cond_dim)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)
        self.output_projection = nn.Conv1d(residual_channels, 1, 3)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        assert t.shape[0] == cond.shape[0] == x.shape[0]
        assert t.min() >= 0 and t.max() <= 1
        assert t.ndim == 1
        # print('------------- Inputs --------------')
        # print(f'x {x.shape} \n t {t.shape} \n cond {cond.shape}')
        
        x = self.input_projection(x)
        x = F.leaky_relu(x, 0.4)

        t = t.unsqueeze(1)
        diffusion_step = self.diffusion_embedding(t)
        cond_up = self.cond_upsampler(cond)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)
        return x
