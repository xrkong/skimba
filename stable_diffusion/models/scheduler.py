from torch import nn
import torch
from typing import List
from dataclasses import dataclass, field

from stable_diffusion.dataclass import BaseDataclass


@dataclass
class DDPMConfig(BaseDataclass):
    noise_schedule: str = field(
        default="linear",
        metadata={
            "help": "Noise schedule type.",
            "choices": ["linear", "cosine", "cubic"],
        },
    )
    noise_steps: int = field(default=1000, metadata={"help": "Number of noise steps."})
    beta_start: float = field(
        default=1e-4, metadata={"help": "Starting value of beta."}
    )
    beta_end: float = field(default=0.02, metadata={"help": "Ending value of beta."})


class DDPMScheduler:
    # @staticmethod
    # def add_ddpm_args(parser):
    #     noise_group = parser.add_argument_group("ddpm")
    #     noise_group.add_argument(
    #         "--noise_schedule",
    #         type=str,
    #         default="linear",
    #         choices=["linear", "cosine", "cubic"],
    #     )
    #     noise_group.add_argument(
    #         "--noise_steps",
    #         type=int,
    #         default=1000,
    #     )
    #     noise_group.add_argument(
    #         "--beta_start",
    #         type=float,
    #         default=1e-4,
    #     )
    #     noise_group.add_argument(
    #         "--beta_end",
    #         type=float,
    #         default=0.02,
    #     )
    #     return noise_group

    def __init__(self, noise_schedule='linear', noise_steps= 1000, beta_start= 1e-4,beta_end=0.02):
        self.noise_schedule = noise_schedule
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        """modified from [labmlai - DDPMSampler](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/stable_diffusion/sampler/ddpm.py)"""
        super().__init__()
        self.noise_steps: int = self.noise_steps
        self.noise_time_steps: torch.Tensor[List[int]] = torch.arange(
            self.noise_steps - 1, -1, -1
        )
        self.betas = self.prepare_linear_noise_schedule(self.beta_start, self.beta_end)
        self.alpha = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alpha, dim=0)
        #  $\bar\alpha_{t-1}$
        alpha_bar_prev = torch.cat(
            [self.alphas_cumprod.new_tensor([1.0]), self.alphas_cumprod[:-1]]
        )
        # $\sqrt{\bar\alpha}$
        self.sqrt_alpha_bar = self.alphas_cumprod**0.5
        # $\sqrt{1 - \bar\alpha}$
        self.sqrt_1m_alpha_bar = (1.0 - self.alphas_cumprod) ** 0.5
        # $\frac{1}{\sqrt{\bar\alpha_t}}$
        self.sqrt_recip_alpha_bar = self.alphas_cumprod**-0.5
        # $\sqrt{\frac{1}{\bar\alpha_t} - 1}$
        self.sqrt_recip_m1_alpha_bar = (1 / self.alphas_cumprod - 1) ** 0.5
        # $\frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t$
        variance = self.betas * (1.0 - alpha_bar_prev) / (1.0 - self.alphas_cumprod)
        # Clamped log of $\tilde\beta_t$
        self.log_var = torch.log(torch.clamp(variance, min=1e-20))
        # $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$
        self.mean_x0_coef = (
            self.betas * (alpha_bar_prev**0.5) / (1.0 - self.alphas_cumprod)
        )
        # $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}$
        self.mean_xt_coef = (
            (1.0 - alpha_bar_prev)
            * ((1 - self.betas) ** 0.5)
            / (1.0 - self.alphas_cumprod)
        )

    def prepare_linear_noise_schedule(self, beta_start, beta_end):
        """
        Returns the linear noise schedule
        """
        return torch.linspace(beta_start, beta_end, self.noise_steps)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        r"""
        sample x_t from q(x_t|x_0), where
        `q(x_t|x_0) = N(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t)I)`
        Modified from [Huggingface/Diffusers - scheduling_ddpm.py](https://github.com/huggingface/diffusers/blob/67cf0445ef48b1f913b90ce0025ac0c75673e32e/src/diffusers/schedulers/scheduling_ddpm.py#L419)

        Args:
            - original_samples (torch.Tensor):
                  x_0, origin latent vector, shape=`[batch, channels, height, width]`
            - noise (torch.Tensor):
                  random noise, shape=`[batch, channels, height, width]`
            - timesteps (torch.Tensor):
                  time step to add noise, shape=`[batch,]`
        Returns:
            - noised latent vector (torch.Tensor):
                    x_t, shape=`[batch, channels, height, width]`
        """
        assert (
            timesteps.max().item() < self.noise_steps
        ), f"timesteps({timesteps}) should be less than {self.noise_steps}"
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    @torch.no_grad()
    def step(
        self,
        pred_noise: torch.Tensor,
        x_t: torch.Tensor,
        time_step: int,
        repeat_noise: bool = False,
        scale_factor: float = 1.0,
    ):
        """
        predict x_t from
        Sample 𝒙_{𝒕-1} from 𝒑_θ(𝒙_{𝒕-1} | 𝒙_𝒕) i.e. decode one step
        Modified from [labmlai - DDPMSampler](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/stable_diffusion/sampler/ddpm.py)

        Args:
            - pred_noise (torch.Tensor):
                  predicted noise from U-Net model = ε_θ(x_t, t, condition)
            - x_t (torch.Tensor):
                  noised latent at timestep=t, shape=`[batch_size, channels, height, width]`
            - time_step (int):
                  integer of timestep=t
            - repeat_noise (bool, optional):
                  whether use the same noise for all items in batch. Default: `False`.
            - scale_factor (float, optional):
                  scale_factor of noise. Default: `1.`.
        """
        assert (
            time_step < self.noise_steps
        ), f"timesteps({time_step}) should be less than {self.noise_steps}"
        bsz = x_t.shape[0]

        # 1 / (\sqrt{\bar\alpha_t})
        # sqrt_recip_alpha_bar = x_t.new_full(
        #     (bsz, 1, 1, 1), self.sqrt_recip_alpha_bar[time_step]
        # )
        sqrt_recip_alpha_bar = x_t.new_full(
            (bsz, 1, 1, 1, 1), self.sqrt_recip_alpha_bar[time_step]
        )
        # $\sqrt{\frac{1}{\bar\alpha_t} - 1}$
        # sqrt_recip_m1_alpha_bar = x_t.new_full(
        #     (bsz, 1, 1, 1), self.sqrt_recip_m1_alpha_bar[time_step]
        # )
        sqrt_recip_m1_alpha_bar = x_t.new_full(
            (bsz, 1, 1, 1, 1), self.sqrt_recip_m1_alpha_bar[time_step]
        )

        # Calculate $x_0$ with current $\epsilon_\theta$
        # Eq (15) from DDPM paper: https://arxiv.org/pdf/2006.11239.pdf
        # $$x_0 = \frac{1}{\sqrt{\bar\alpha_t}} x_t -  \Big(\sqrt{\frac{1}{\bar\alpha_t} - 1}\Big)\epsilon_\theta$$
        a = sqrt_recip_alpha_bar * x_t
        b = sqrt_recip_m1_alpha_bar * pred_noise


        x0 = sqrt_recip_alpha_bar * x_t - sqrt_recip_m1_alpha_bar * pred_noise

        # $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$
        # mean_x0_coef = x_t.new_full((bsz, 1, 1, 1), self.mean_x0_coef[time_step])
        mean_x0_coef = x_t.new_full((bsz, 1, 1, 1,1), self.mean_x0_coef[time_step])
        # $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}$
        # mean_xt_coef = x_t.new_full((bsz, 1, 1, 1), self.mean_xt_coef[time_step])
        mean_xt_coef = x_t.new_full((bsz, 1, 1, 1,1), self.mean_xt_coef[time_step])

        # Calculate $\mu_t(x_t, t)$
        # $$\mu_t(x_t, t) = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
        #    + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$
        mean = mean_x0_coef * x0 + mean_xt_coef * x_t
        # $\log \tilde\beta_t$
        # log_var = x_t.new_full((bsz, 1, 1, 1), self.log_var[time_step])
        log_var = x_t.new_full((bsz, 1, 1, 1,1), self.log_var[time_step])

        # Do not add noise when $t = 1$ (final step sampling process).
        # Note that `step` is `0` when $t = 1$)
        if time_step == 0:
            noise = torch.zeros(x_t.shape)
        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x_t.shape[1:]))
        # Different noise for each sample
        else:
            noise = torch.randn(x_t.shape)

        # Multiply noise by the temperature
        noise = noise.to(mean.device, dtype=mean.dtype) * scale_factor
        # Sample from,
        # $$p_\theta(x_{t-1} | x_t) = \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \tilde\beta_t \mathbf{I} \big)$$
        x_prev = mean + (0.5 * log_var).exp() * noise
        # return x_0 if we try to get x_0 directly from x_t, but it makes distortion more difficult to evaluate,
        # see Eq (5) from DDPM paper: https://arxiv.org/pdf/2006.11239.pdf
        return (
            x_prev,
            x0
        )
