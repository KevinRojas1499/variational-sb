from typing import Union
import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gluonts.core.component import validated
from gluonts.itertools import select

from module import DiffusionModel

class DiffusionLightningModule(pl.LightningModule):
    @validated()
    def __init__(
        self,
        sde : str,
        damp_coef : float,
        beta_max : float,
        dsm_warm_up : int,
        dsm_cool_down : int,
        forward_opt_steps: Union[int, float],
        backward_opt_steps: Union[int, float],
        model_kwargs: dict,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.model = DiffusionModel(sde=sde, damp_coef=damp_coef, beta_max=beta_max, dsm_warm_up=dsm_warm_up,
                                    dsm_cool_down=dsm_cool_down,forward_opt_steps=forward_opt_steps,
                                    backward_opt_steps=backward_opt_steps,**model_kwargs)

        self.sde = sde  
        self.damp_coef = damp_coef
        self.beta_max = beta_max
        self.dsm_warm_up = dsm_warm_up  
        self.dsm_cool_down = dsm_cool_down  
        self.forward_opt_steps = forward_opt_steps
        self.backward_opt_steps = backward_opt_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.inputs = self.model.describe_inputs()
        self.example_input_array = self.inputs.zeros()

        # Will change during training
        self.step = 0
        self.direction = 'backward'

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx: int):

        f_opt, b_opt = self.optimizers()
        f_lr, b_lr = self.lr_schedulers()

        f_opt.zero_grad()
        b_opt.zero_grad()

        train_loss = self.model.loss(
            **select(self.inputs, batch),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
            step=self.step,
        ).mean()
        self.step += 1
        train_loss.backward()
        self.clip_gradients(f_opt, gradient_clip_val=10, gradient_clip_algorithm="norm")
        
        f_opt.step()
        b_opt.step()

        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        val_loss = self.model.loss(
            **select(self.inputs, batch),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
        ).mean()

        self.log("val_loss", val_loss, prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        forward_opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        backward_opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        forward_lr_scheduler = ReduceLROnPlateau(
            optimizer=forward_opt,
            mode="min",
            factor=0.5,
            patience=self.patience,
            verbose=True,
        )
        backward_lr_scheduler = ReduceLROnPlateau(
            optimizer=backward_opt,
            mode="min",
            factor=0.5,
            patience=self.patience,
            verbose=True,
        )
        return (
            {
                'optimizer': forward_opt,
                'lr_scheduler': forward_lr_scheduler,
            },
            {
                'optimizer': backward_opt,
                'lr_scheduler': backward_lr_scheduler,
            },
        )
