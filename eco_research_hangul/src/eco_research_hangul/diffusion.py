from __future__ import annotations

from dataclasses import dataclass

import torch


def _extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 64
    beta_start: float = 1e-4
    beta_end: float = 0.02


class DiffusionSchedule:
    def __init__(
        self,
        timesteps: int = 64,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | str = "cpu",
    ) -> None:
        self.timesteps = int(timesteps)
        self.device = torch.device(device)
        self.betas = torch.linspace(beta_start, beta_end, self.timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, x0.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def predict_eps_from_x0(self, xt: torch.Tensor, timesteps: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, xt.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, xt.shape)
        return (xt - sqrt_alpha * x0) / sqrt_one_minus.clamp_min(1e-8)

    @torch.no_grad()
    def p_sample(self, model, xt: torch.Tensor, condition: torch.Tensor, timestep: int, prediction_type: str = "epsilon") -> torch.Tensor:
        t = torch.full((xt.shape[0],), timestep, device=xt.device, dtype=torch.long)
        predicted = model(xt, condition, t)
        if prediction_type == "x0":
            predicted_noise = self.predict_eps_from_x0(xt, t, predicted.clamp(-1.0, 1.0))
        elif prediction_type == "epsilon":
            predicted_noise = predicted
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")
        beta_t = _extract(self.betas, t, xt.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        sqrt_recip_alpha = _extract(self.sqrt_recip_alphas, t, xt.shape)
        model_mean = sqrt_recip_alpha * (xt - beta_t * predicted_noise / sqrt_one_minus.clamp_min(1e-8))
        if timestep == 0:
            return model_mean
        variance = _extract(self.posterior_variance, t, xt.shape)
        return model_mean + torch.sqrt(variance.clamp_min(1e-20)) * torch.randn_like(xt)

    @torch.no_grad()
    def sample_loop(
        self,
        model,
        condition: torch.Tensor,
        image_size: int,
        channels: int = 1,
        steps: int | None = None,
        prediction_type: str = "epsilon",
    ) -> torch.Tensor:
        batch = condition.shape[0]
        x = torch.randn(batch, channels, image_size, image_size, device=condition.device)
        if steps is None or steps >= self.timesteps:
            schedule = list(range(self.timesteps - 1, -1, -1))
        else:
            schedule = torch.linspace(self.timesteps - 1, 0, steps, device=condition.device).round().long().tolist()
        for timestep in schedule:
            x = self.p_sample(model, x, condition, int(timestep), prediction_type=prediction_type)
        return x.clamp(-1.0, 1.0)
