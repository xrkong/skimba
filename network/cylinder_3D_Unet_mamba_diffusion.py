from torch import nn
import torch
import torch.nn.functional as F

REGISTERED_MODELS_CLASSES = {}
pytorch_device = torch.device('cuda:0')

def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]


@register_model
class cylinder_asym(nn.Module):
    def __init__(self,
                 model_part,
                 sparse_shape,
                 channels_list,
                 num_input_features,
                 init_size,
                 voxel_channel
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.sparse_shape = sparse_shape
        self.model_part = model_part

    def forward(self, batch_size, val_VAE_features_change, val_partial_features_change, val_image_features_change, train=True):

        if train == True:
            noise_complete = torch.randn(val_VAE_features_change.shape).to(pytorch_device)
            ###------------------------------------sample a random timestep for each voxel scene-------
            noise_steps = 1000
            timesteps_complete = torch.randint(noise_steps, (val_VAE_features_change.shape[0],)).to(pytorch_device,dtype=torch.long)
            x_t_complete = self.model_part.noise_scheduel_ddpm.add_noise(original_samples=val_VAE_features_change,
                                                                         timesteps=timesteps_complete,
                                                                         noise=noise_complete).to(dtype=torch.float32)
            pred_noise = self.model_part.pred_noise(x_t_complete,
                                                    val_partial_features_change,
                                                    val_image_features_change,
                                                    timesteps_complete,
                                                    guidance_scale=3.0,
                                                    train=train)

            return noise_complete, pred_noise
        else:
            noise_complete = torch.randn(val_VAE_features_change.shape).to(pytorch_device)
            ###------------------------------------sample a random timestep for each voxel scene-------
            noise_steps = 1000
            timesteps_complete = torch.randint(noise_steps, (val_VAE_features_change.shape[0],)).to(pytorch_device,dtype=torch.long)
            x_t_complete = self.model_part.noise_scheduel_ddpm.add_noise(original_samples=val_VAE_features_change,
                                                                         timesteps=timesteps_complete,
                                                                         noise=noise_complete).to(dtype=torch.float32)
            pred_noise = self.model_part.pred_noise(x_t_complete,
                                                    val_partial_features_change,
                                                    val_image_features_change,
                                                    timesteps_complete,
                                                    guidance_scale=3.0,
                                                    train=train)

            loss_val = F.mse_loss(pred_noise.float(), noise_complete.float(), reduction="mean")
            num_inference_steps = 100
            x_yuan_recover = self.model_part.sample(noise_complete,
                                                    val_partial_features_change,
                                                    val_image_features_change,
                                                    3.0,
                                                    num_inference_steps,
                                                    False)
            recon_voxel = self.model_part.autoencoder.decode(x_yuan_recover)

            return loss_val, recon_voxel

