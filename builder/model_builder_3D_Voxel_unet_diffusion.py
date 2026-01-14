from network.cylinder_3D_Unet_mamba_diffusion import get_model_class
from stable_diffusion.models.autoencoder import AutoEncoderKL
from stable_diffusion.models.skimba import SegMamba
from stable_diffusion.models.latent_diffusion import LatentDiffusion
from diffusers import DDPMScheduler
from diffusers import DDIMScheduler

def build(model_config):
    output_shape = model_config['output_shape']
    num_input_features = model_config['num_input_features']
    init_size = model_config['init_size']
    channels_list = model_config['channels_list']
    latent_channels = model_config['latent_channels']
    in_channels = model_config['in_channels']
    out_channels = model_config['out_channels']
    autoencoder_num_res_blocks = model_config['autoencoder_num_res_blocks']
    autoencoder_channels_list = model_config['autoencoder_channels_list']
    auto_groups = model_config['auto_groups']
    voxel_channel = model_config['voxel_channel']

    unet = SegMamba()
    autoencoder = AutoEncoderKL(in_channels=in_channels, latent_channels=latent_channels, out_channels=out_channels,
                                autoencoder_num_res_blocks=autoencoder_num_res_blocks,
                                autoencoder_channels_list=autoencoder_channels_list, groups=auto_groups)

    SchedulerDDPM = DDPMScheduler(num_train_timesteps = 1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', clip_sample=False, steps_offset=1)
    SchedulerDDIM = DDIMScheduler(num_train_timesteps = 1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', clip_sample=False, steps_offset=1, set_alpha_to_one=False)
    # 0.00085   0.012
    model_part = LatentDiffusion(unet, autoencoder, SchedulerDDPM, SchedulerDDIM)

    model = get_model_class(model_config["model_architecture"])(model_part = model_part,
                                                                sparse_shape=output_shape,
                                                                channels_list = channels_list,
                                                                num_input_features =num_input_features,
                                                                init_size = init_size,
                                                                voxel_channel = voxel_channel
                                                                )
    return model
