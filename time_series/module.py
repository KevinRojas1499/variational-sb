from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor

from tqdm import tqdm
from gluonts.core.component import validated
from gluonts.itertools import prod
from gluonts.model import Input, InputSpec
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.scaler import MeanScaler, NOPScaler, Scaler, StdScaler
from gluonts.torch.util import repeat_along_dim, unsqueeze_expand
from pts.util import lagged_sequence_values


from epsilon_theta import EpsilonTheta

# script.py
import sys
import os

# Add the parent directory to the sys.path to ensure it can find hello.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.training_routines import get_routine
import utils.sde_lib as SDEs
from utils.models import MatrixTimeEmbedding
from utils.model_utils import get_preconditioned_model
from utils.misc import dotdict

class DiffusionModel(nn.Module):
    """
    Module implementing the T2TSBModel model.

    Parameters
    ----------
    freq
        String indicating the sampling frequency of the data to be processed.
    context_length
        Length of the RNN unrolling prior to the forecast date.
    prediction_length
        Number of time points to predict.
    num_feat_dynamic_real
        Number of dynamic real features that will be provided to ``forward``.
    num_feat_static_real
        Number of static real features that will be provided to ``forward``.
    num_feat_static_cat
        Number of static categorical features that will be provided to
        ``forward``.
    cardinality
        List of cardinalities, one for each static categorical feature.
    embedding_dimension
        Dimension of the embedding space, one for each static categorical
        feature.
    num_layers
        Number of layers in the RNN.
    hidden_size
        Size of the hidden layers in the RNN.
    dropout_rate
        Dropout rate to be applied at training time.
    lags_seq
        Indices of the lagged observations that the RNN takes as input. For
        example, ``[1]`` indicates that the RNN only takes the observation at
        time ``t-1`` to produce the output for time ``t``; instead,
        ``[1, 25]`` indicates that the RNN takes observations at times ``t-1``
        and ``t-25`` as input.
    scaling
        Whether to apply mean scaling to the observations (target).
    default_scale
        Default scale that is applied if the context length window is
        completely unobserved. If not set, the scale in this case will be
        the mean scale in the batch.
    num_parallel_samples
        Number of samples to produce when unrolling the RNN in the prediction
        time range.
    """

    @validated()
    def __init__(
        self,
        sde : str,
        damp_coef : float,
        beta_max : float,
        dsm_warm_up : int,
        dsm_cool_down : int,
        forward_opt_steps: int,
        backward_opt_steps: int,
        freq: str,
        context_length: int,
        prediction_length: int,
        n_timestep: int = 150,
        input_size: int = 1,
        num_feat_dynamic_real: int = 1,
        num_feat_static_real: int = 1,
        num_feat_static_cat: int = 1,
        cardinality: List[int] = [1],
        embedding_dimension: Optional[List[int]] = None,
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        lags_seq: Optional[List[int]] = None,
        scaling: Optional[str] = "mean",
        default_scale: float = 0.0,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()

        assert num_feat_dynamic_real > 0
        assert num_feat_static_real > 0
        assert num_feat_static_cat > 0
        assert len(cardinality) == num_feat_static_cat
        assert (
            embedding_dimension is None
            or len(embedding_dimension) == num_feat_static_cat
        )

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.input_size = input_size

        self.n_timestep = n_timestep

        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )
        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.lags_seq = [lag - 1 for lag in self.lags_seq]
        self.num_parallel_samples = num_parallel_samples
        self.past_length = self.context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality, embedding_dims=self.embedding_dimension
        )
        if scaling == "mean":
            self.scaler: Scaler = MeanScaler(
                dim=1, keepdim=True, default_scale=default_scale
            )
        elif scaling == "std":
            self.scaler: Scaler = StdScaler(dim=1, keepdim=True)
        else:
            self.scaler: Scaler = NOPScaler(dim=1, keepdim=True)

        self.rnn_input_size = (
            self.input_size * len(self.lags_seq) + self._number_of_features
        )

        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        
        self.sde = SDEs.get_sde(sde, beta_max=beta_max)
        augmented = self.sde.is_augmented 
        self.backward_net = EpsilonTheta(
            in_channels=2 if augmented else 1,
            target_dim=input_size,
            cond_dim=hidden_size,
            interval=self.n_timestep,
        )
        self.backward_net = get_preconditioned_model(self.backward_net,self.sde)
        self.forward_net = MatrixTimeEmbedding([1,input_size], gamma=self.sde.gamma if augmented else None, is_augmented=augmented, under_damp_coeff=damp_coef) 

        self.sde.backward_score = self.backward_net
        if hasattr(self.sde,'forward_score'):
            self.sde.forward_score = self.forward_net
        self.routine = get_routine(dotdict({
            'sde': sde,
            'dsm_warm_up': dsm_warm_up,
            'num_iters' : 2500, # TODO : Not hardcode this value, it is given under the assumption of 50 epochs
            'dsm_cool_down': dsm_cool_down,
            'backward_opt_steps': backward_opt_steps,
            'forward_opt_steps': forward_opt_steps
        }), 2500, self.sde,self.sde)

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "feat_static_cat": Input(
                    shape=(batch_size, self.num_feat_static_cat), dtype=torch.long
                ),
                "feat_static_real": Input(
                    shape=(batch_size, self.num_feat_static_real), dtype=torch.float
                ),
                "past_time_feat": Input(
                    shape=(batch_size, self._past_length, self.num_feat_dynamic_real),
                    dtype=torch.float,
                ),
                "past_target": Input(
                    shape=(batch_size, self._past_length)
                    if self.input_size == 1
                    else (batch_size, self._past_length, self.input_size),
                    dtype=torch.float,
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self._past_length)
                    if self.input_size == 1
                    else (batch_size, self._past_length, self.input_size),
                    dtype=torch.float,
                ),
                "future_time_feat": Input(
                    shape=(
                        batch_size,
                        self.prediction_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
            },
            zeros_fn=torch.zeros,
        )

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + self.input_size * 2  # the log(scale) and log1p(abs(loc))
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def prepare_rnn_input(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,]:
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
    ) -> Tuple[
        Tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
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

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Invokes the model on input data, and produce outputs future samples.

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
            (Optional) tensor of dynamic real features in the past,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        num_parallel_samples
            How many future samples to produce.
            By default, self.num_parallel_samples is used.
        """
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        loc, scale, rnn_output, static_feat, state = self.unroll_lagged_rnn(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat[:, :1],
        )

        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_static_feat = static_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        ).unsqueeze(dim=1)
        repeated_past_target = (
            past_target.repeat_interleave(repeats=num_parallel_samples, dim=0)
            - repeated_loc
        ) / repeated_scale
        repeated_time_feat = future_time_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        repeated_state = [
            s.repeat_interleave(repeats=num_parallel_samples, dim=1) for s in state
        ]
        repeated_outputs = rnn_output.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )

        # sample
        sampling_shape = [repeated_past_target.shape[0], 2 if self.sde.is_augmented else 1, repeated_past_target.shape[-1]]
        next_sample = self.sde.sample(shape=sampling_shape,
                device=repeated_past_target[:, -1:, ...].device,
            cond=repeated_outputs[:, -1:, ...])[0]
        
        future_samples = [repeated_scale * next_sample + repeated_loc]

        for k in tqdm(range(1, self.prediction_length)):
            next_features = torch.cat(
                (repeated_static_feat, repeated_time_feat[:, k : k + 1]),
                dim=-1,
            )
            next_lags = lagged_sequence_values(
                self.lags_seq, repeated_past_target, next_sample, dim=1
            )
            rnn_input = torch.cat((next_lags, next_features), dim=-1)

            repeated_outputs, repeated_state = self.rnn(rnn_input, repeated_state)

            repeated_past_target = torch.cat(
                (repeated_past_target, next_sample), dim=1
            )

            next_sample = self.sde.sample(shape=sampling_shape,
                device=repeated_past_target[:, -1:, ...].device,
                cond=repeated_outputs[:, -1:, ...])[0]
            future_samples.append(repeated_scale * next_sample + repeated_loc)

        future_samples_concat = torch.cat(future_samples, dim=1).reshape(
            (-1, num_parallel_samples, self.prediction_length, self.input_size)
        )

        return future_samples_concat.squeeze(-1)

    def get_loss_values(
        self,
        rnn_outputs: Tensor,
        target: Tensor,
        observed_values: Tensor,
        step: int,
    ):
        loss = self.routine.get_loss(step,
                data=target.reshape(-1,1,target.shape[-1]),
                cond=rnn_outputs.reshape(-1,1,rnn_outputs.shape[-1]))
        return loss

    def loss(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        future_only: bool = True,
        aggregate_by=torch.mean,
        step: int = None,
    ) -> torch.Tensor:
        extra_dims = len(future_target.shape) - len(past_target.shape)
        extra_shape = future_target.shape[:extra_dims]

        repeats = prod(extra_shape)
        feat_static_cat = repeat_along_dim(feat_static_cat, 0, repeats)
        feat_static_real = repeat_along_dim(feat_static_real, 0, repeats)
        past_time_feat = repeat_along_dim(past_time_feat, 0, repeats)
        past_target = repeat_along_dim(past_target, 0, repeats)
        past_observed_values = repeat_along_dim(past_observed_values, 0, repeats)
        future_time_feat = repeat_along_dim(future_time_feat, 0, repeats)

        future_target_reshaped = future_target.reshape(
            -1, *future_target.shape[extra_dims + 1 :]
        )
        future_observed_reshaped = future_observed_values.reshape(
            -1, *future_observed_values.shape[extra_dims + 1 :]
        )

        loc, scale, rnn_outputs, _, _ = self.unroll_lagged_rnn(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target_reshaped,
        )
        
        if future_only:
            sliced_rnn_outputs = rnn_outputs[:, -self.prediction_length :]
            observed_values = (
                future_observed_reshaped.all(-1)
                if future_observed_reshaped.ndim == 3
                else future_observed_reshaped
            )

            future_target_reshaped = (future_target_reshaped - loc) / scale

            loss = self.get_loss_values(
                rnn_outputs=sliced_rnn_outputs,
                target=future_target_reshaped,
                observed_values=observed_values,
                step=step,
            )
        else:
            context_target = past_target[:, -self.context_length + 1 :, ...]
            target = torch.cat((context_target, future_target_reshaped), dim=1)
            context_observed = past_observed_values[:, -self.context_length + 1 :, ...]
            observed_values = torch.cat(
                (context_observed, future_observed_reshaped), dim=1
            )
            observed_values = (
                observed_values.all(-1)
                if observed_values.ndim == 3
                else observed_values
            )

            target = (target - loc) / scale

            loss = self.get_loss_values(
                rnn_outputs=rnn_outputs,
                target=target,
                observed_values=observed_values,
                step=step,
            )

        return loss
