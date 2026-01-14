from typing import Optional
import torch
from torch import nn
from tqdm import tqdm

class LatentDiffusion(nn.Module):
    def __init__(
        self,
        unet,
        autoencoder,
        noise_scheduel_ddpm,
        noise_scheduel_ddim
    ):
        """main class"""
        super().__init__()
        self.unet = unet
        self.autoencoder = autoencoder
        self.noise_scheduel_ddpm = noise_scheduel_ddpm
        self.noise_scheduel_ddim = noise_scheduel_ddim

    def pred_noise(
        self,
        noised_sample: torch.Tensor,
        context_emb_partial: torch.Tensor,
        context_emb_image: torch.Tensor,
        time_step: torch.Tensor,
        guidance_scale: float = 1.0,
        train: bool=True
    ):
        """
        predicts the noise added on latent vector at time step=t

        Args:
            - x (torch.Tensor):
                  noised latent vector, shape = `[batch, channels, height, width]`
            - time_steps (torch.Tensor):
                  time steps, shape = `[batch]`
            - context_emb (torch.Tensor):
                  conditional embedding, shape = `[batch, seq_len, d_model]`

        Returns:
            - pred noise (torch.Tensor):
                  predicted noise, shape = `[batch, channels, height, width]`
        """
        do_classifier_free_guidance = guidance_scale > 1
        if not do_classifier_free_guidance:
            b, c, l, w, h = context_emb_partial.shape
            context_emb_partial = context_emb_partial.flatten(start_dim=2).permute(0, 2, 1)
            context_emb_image = context_emb_image.flatten(start_dim=2).permute(0, 2, 1)
            context_emb_all = torch.cat((context_emb_partial, context_emb_image), dim=2)
            context_emb_all = context_emb_all.view(b, 16, l, w, h)
            prediced_noise = self.unet(noised_sample, context_emb_all, time_step)
            return prediced_noise
        if train==True:
            t_in = torch.cat([time_step] * 2)
            x_in = torch.cat([noised_sample] * 2)
            context_emb_all = torch.cat((context_emb_partial, context_emb_image), dim=1)
            mask = torch.randn_like(context_emb_all) > 0.1
            context_emb_all_final = context_emb_all * mask
            c_in_all = torch.cat([context_emb_all_final, context_emb_all], dim=0)
            pred_noise_uncond, pred_noise_cond = torch.chunk(self.unet(x_in, c_in_all, t_in), 2, dim=0)
            return pred_noise_uncond
        else:
            t_in = torch.cat([time_step] * 2)
            x_in = torch.cat([noised_sample] * 2)
            context_emb_all = torch.cat((context_emb_partial, context_emb_image), dim=1)
            c_in_empty = torch.zeros(context_emb_all.shape).to('cuda:0')
            c_in_all = torch.cat([c_in_empty, context_emb_all], dim=0)
            pred_noise_uncond, pred_noise_cond = torch.chunk(self.unet(x_in, c_in_all, t_in), 2, dim=0)
            return pred_noise_cond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

    def sample(
        self,
        noised_sample: torch.Tensor,
        partial_features_change: torch.Tensor,
        image_features_change: torch.Tensor,
        guidance_scale: float = 7.5,
        # repeat_noise: bool = False,
        # scale_factor: float = 1.0,
        # time_steps: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        train: bool=False
    ):
        """
        Sample loop to get x_0 given x_t and conditional embedding

        Args:
            - noised_sample (torch.Tensor):
                  original x_t of shape=`[batch, channels, height, width]`
            - context_emb (torch.Tensor):
                  conditional embedding of shape=`[batch, seq_len, context_dim]`
            - guidence_scale (float, optional):
                  scale used for classifer free guidance. Default: `7.5`.
            - repeat_noise (bool, optional):
                  whether use the same noise in a batch duuring each p_sample. Default: `False`.
            - scale_factor (float, optional):
                  scaling factor of noise. Default: `1.0`.

        Returns:
            - x_0 (torch.Tensor):
                  denoised latent x_0 of shape=`[batch, channels, height, width]`
        """
        bsz = noised_sample.shape[0]

        # Get x_T
        x = noised_sample

        # Time steps to sample at $T - t', T - t' - 1, \dots, 1$
        # Sampling loop
        pytorch_device = torch.device('cuda:0')
        self.noise_scheduel_ddim.set_timesteps(num_inference_steps, pytorch_device)
        time_steps = self.noise_scheduel_ddim.timesteps
        # progress_bar = tqdm(total=len(time_steps))
        progress_bar = tqdm(total=len(time_steps))
        for i, j in enumerate(time_steps):
            # fill time step t from int to tensor of shape=`[batch]`
            time_step = x.new_full((bsz,), j, dtype=torch.long)
            pred_noise = self.pred_noise(
                noised_sample=x,
                time_step=time_step,
                context_emb_partial=partial_features_change,
                context_emb_image = image_features_change,
                guidance_scale=guidance_scale,
                train=train
            )
            x = self.noise_scheduel_ddim.step(
                model_output=pred_noise,
                timestep=j,
                sample=x)
            x=x['prev_sample']
            progress_bar.update(1)
        progress_bar.close()
        return x