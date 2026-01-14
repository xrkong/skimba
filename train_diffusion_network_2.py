import os
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from dataloader.pc_dataset_gai import get_eval_mask, unpack
from builder import data_builder_autoencoder, model_builder_3D_Voxel_unet_diffusion, loss_builder
from config.config import load_config_data
from torch.utils.tensorboard import SummaryWriter
import warnings
from utils.np_ioueval import iouEval
import yaml
from warmup_lr import *
import torch.nn.functional as F
from collections import OrderedDict

def freeze_model(model, to_free_dict):
    for name, param in model.named_parameters():
        if name in to_free_dict:
            param.requires_grad = False
        else:
            pass
    return model

def main(args):
    writer = SummaryWriter('./logs')

    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    pytorch_device = torch.device('cuda:0')

    with open("/home/Software/AAAI_Code/3D_voxel_diffusion_plane/config/label_mapping/semantic-kitti.yaml", 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    learning_map_inv = semkittiyaml["learning_map_inv"]
    maxkey = max(learning_map_inv.keys())
    remap_lut_First = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut_First[list(learning_map_inv.keys())] = list(learning_map_inv.values())
    config_path = args.config_path
    configs = load_config_data(config_path)
    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    my_model = model_builder_3D_Voxel_unet_diffusion.build(model_config)
    model_load_path += '0.pth'
    total_params = sum(p.numel() for p in my_model.parameters())
    print(f"总参数量: {total_params}")
    if os.path.exists(model_load_path):
        print('Load model from: %s' % model_load_path)
        checkpoint = torch.load(model_load_path)
        my_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No existing model, training model from scratch...')

    model_load_path_VAE = '/home/Software/Code_Li_bev_one_to_one/model_save_path/Complete_64_64_8_8_2_0.0003_0.001_dense/' + 'best_24_99.46710019208265.pth'
    if os.path.exists(model_load_path_VAE):
        print('Load model from: %s' % model_load_path_VAE)
        pre_state_dict = torch.load(model_load_path_VAE,map_location=pytorch_device)
        subset_state_dict_1 = {}
        for name, param in pre_state_dict.items():
            if 'model_part' in name:
                name = name.replace('model_part.', '')
                subset_state_dict_1[name] = param
        subset_state_dict_1 = OrderedDict(subset_state_dict_1)
        my_model.model_part.autoencoder.load_state_dict(subset_state_dict_1, strict=True)
        for param in my_model.model_part.autoencoder.parameters():
            param.requires_grad = False
    else:
        print('No existing model, training model from scratch...')

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    print(model_save_path)

    my_model.to(pytorch_device)
    optimizer = optim.AdamW(my_model.parameters(), lr=train_hypers["learning_rate"], weight_decay=1e-4)

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True, num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader, val_pt_dataset = data_builder_autoencoder.build(dataset_config,
                                                                                  train_dataloader_config,
                                                                                  val_dataloader_config,
                                                                                  grid_size=grid_size,
                                                                                  use_tta=False,
                                                                                  use_multiscan=True)

    scheduler = WarmupCosineLR(optimizer=optimizer, lr=train_hypers["learning_rate"],
                               warmup_steps=train_hypers['warmup_epochs'] * len(train_dataset_loader), momentum=0.9,
                               max_steps=len(train_dataset_loader) * (train_hypers['max_num_epochs'] - train_hypers['warmup_epochs']))

    # training
    start_epoch = 1
    if os.path.exists(model_load_path):
        print('Load model from: %s' % model_load_path)
        checkpoint = torch.load(model_load_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        print('No existing model, training model from scratch...')

    best_val_miou = 0
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']
    check_epoch = 1
    max_epoch = train_hypers['max_num_epochs']

    # learning map
    with open("config/label_mapping/semantic-kitti.yaml", 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    class_strings = semkittiyaml["labels"]
    class_inv_remap = semkittiyaml["learning_map_inv"]

    bin_file_path = "" #h-w-d-c-bs-lr-kl mathod_noisestep
    if not os.path.exists(bin_file_path):
        os.makedirs(bin_file_path)
    print(bin_file_path)

    while start_epoch < max_epoch+1:
        pbar = tqdm(total=len(train_dataset_loader))
        my_model.train()
        for module in my_model.model_part.autoencoder.modules():
            if isinstance(module, nn.BatchNorm3d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()
        train_loss_mse = []
        for i_iter, (train_GT_features, train_VAE_features, train_partial_features, train_image_features, origin_len, dir_idx, number_idx) in enumerate(train_dataset_loader):  # origin_len
            train_VAE_features_change = train_VAE_features.type(torch.FloatTensor).to(pytorch_device)
            train_partial_features_change = train_partial_features.type(torch.FloatTensor).to(pytorch_device)
            train_image_features_change = train_image_features.type(torch.FloatTensor).to(pytorch_device)

            # forward + backward + optimize
            noise, pred_noise = my_model(
                                                     train_VAE_features_change.shape[0],
                                                     train_VAE_features_change,
                                                     train_partial_features_change,
                                                     train_image_features_change,
                                                     train = True
                                                     )

            ###------------------------------------acquire recover feature --------------------------------------------------#####
            #----mse----
            loss_train = F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")
            print('epoch %d iter %d total train loss %.8f  \n' %(start_epoch,i_iter,loss_train))
            train_loss_mse.append(loss_train.item())
            loss_train.backward()
            optimizer.step()
            scheduler.step()
            if global_iter % 1000 == 0:
                if len(train_loss_mse) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (start_epoch, i_iter, np.mean(train_loss_mse)))
                else:
                    print('loss error')

            optimizer.zero_grad()
            pbar.update(1)
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(train_loss_mse) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (start_epoch, i_iter, np.mean(train_loss_mse)))
                else:
                    print('loss error')

        if start_epoch % check_epoch == 0 or start_epoch == max_epoch:  # and epoch > 0
            my_model.eval()
            val_loss_list = []
            val_reconstruction_loss = []
            evaluator = iouEval(num_class, [])
            with torch.no_grad():
                for i_iter_val, (val_GT_features, val_VAE_features, val_partial_features, val_image_features, origin_len, dir_idx, number_idx) in enumerate(val_dataset_loader):

                    val_VAE_features_change = val_VAE_features.type(torch.FloatTensor).to(pytorch_device)
                    val_partial_features_change = val_partial_features.type(torch.FloatTensor).to(pytorch_device)
                    val_image_features_change = val_image_features.type(torch.FloatTensor).to(pytorch_device)

                    val_GT_features_change = val_GT_features.type(torch.LongTensor).to(pytorch_device)

                    loss_val_mse, recon_voxel = my_model(
                                                        val_VAE_features_change.shape[0],
                                                        val_VAE_features_change,
                                                        val_partial_features_change,
                                                        val_image_features_change,
                                                        train = False
                                                                               )
                    #------------------------calculate the loss in the validation dataset---------------------------------------------#
                    reconstruction_loss = loss_func(recon_voxel.detach(),val_GT_features_change)

                    print('epoch %d iter %d loss_val_mse %.8f reconstruction_loss %.8f \n' % (start_epoch, i_iter_val, loss_val_mse, reconstruction_loss))

                    val_vox_label0 = val_GT_features_change.cpu().detach().numpy()
                    val_vox_label0 = np.squeeze(val_vox_label0)
                    recon_voxel = F.softmax(recon_voxel)
                    predict_labels = torch.argmax(recon_voxel, dim=1)
                    predict_labels = predict_labels.cpu().detach().numpy()
                    predict_labels = np.squeeze(predict_labels)
                    invalid_name = '/home/SSC_CODE/Data/data_odometry_voxel_all/sequences/' + "{:02}".format(dir_idx[0]) + '/voxels/' + "{:06}.invalid".format(number_idx[0])
                    invalid_voxels = unpack(np.fromfile(invalid_name, dtype=np.uint8))  # voxel labels
                    invalid_voxels = invalid_voxels.reshape((256, 256, 32))
                    masks = get_eval_mask(val_vox_label0, invalid_voxels)
                    predict_labels = predict_labels[masks]
                    val_vox_label0 = val_vox_label0[masks]

                    evaluator.addBatch(predict_labels.astype(int), val_vox_label0.astype(int))
                    val_loss_list.append(loss_val_mse.item())
                    val_reconstruction_loss.append(reconstruction_loss.item())
                    if start_epoch % 43 ==0:
                        ###--------------------------------------------save the validation result to visualization--------------------------------------###

                        fout_pred_voxel_path = os.path.join(bin_file_path, "sequences_{}".format(start_epoch), "{:02}".format(dir_idx[0]),'pred')
                        if not os.path.exists(fout_pred_voxel_path):
                            os.makedirs(fout_pred_voxel_path)

                        fout_pred_voxel = os.path.join(fout_pred_voxel_path, "{:06}.label".format(number_idx[0]))
                        recon_voxel_occupy = torch.argmax(recon_voxel, dim=1)
                        recon_voxel_1 = recon_voxel_occupy
                        recon_voxel_1 = recon_voxel_1.cpu().numpy()

                        pred = recon_voxel_1.astype(np.uint32)
                        pred = pred.reshape((-1))
                        upper_half = pred >> 16  # get upper half for instances
                        lower_half = pred & 0xFFFF  # get lower half for semantics
                        lower_half = remap_lut_First[lower_half]  # do the remapping of semantics
                        pred = (upper_half << 16) + lower_half  # reconstruct full label
                        pred = pred.astype(np.uint32)
                        final_preds = pred.astype(np.uint16)
                        final_preds.tofile(fout_pred_voxel)

            print('Validation per class iou: ')
            _, class_jaccard = evaluator.getIoU()
            m_jaccard = class_jaccard[1:].mean()
            iou = class_jaccard
            val_miou = m_jaccard * 100
            ignore = [0]
            outpath = model_save_path + 'output_{}.txt'.format(start_epoch)
            # print also classwise
            for i, jacc in enumerate(class_jaccard):
                if i not in ignore:
                    print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                        i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))
                    with open(outpath, 'a', encoding='utf-8') as file:
                        file.write('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(i=i, class_str=class_strings[
                            class_inv_remap[i]], jacc=jacc) + "\n")
            # compute remaining metrics.
            conf = evaluator.get_confusion()
            acc_completion = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0])
            print('Current val completion iou is %.3f and Current val miou is %.3f' % (acc_completion, val_miou))
            with open(outpath, 'a', encoding='utf-8') as file:
                file.write('Current val completion iou is %.3f and Current val miou is %.3f' % (acc_completion, val_miou) + "\n")

            del val_GT_features, val_VAE_features, val_partial_features, val_image_features, val_VAE_features_change, val_partial_features_change, val_image_features_change, val_GT_features_change
                #val_vox_label, point_label_tensor, point_label_tensor_1

            if best_val_miou < val_miou:
                best_val_miou = val_miou
                # model_save_name = model_save_path + ('iou%.4f_epoch%d.pth' % (val_loss, epoch))
                model_save_name = model_save_path + ('{}_{}_{}.pth'.format('best', start_epoch, best_val_miou))
                torch.save(my_model.state_dict(), model_save_name)
                model_save_name_protect = model_save_path + ('{}_{}.pth'.format('protect', start_epoch))
                torch.save({'epoch': start_epoch,
                            'model_state_dict': my_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'Loss': best_val_miou},
                           model_save_name_protect)
            print('Current val miou is %.3f while the best val miou is %.3f' % (val_miou, best_val_miou))
            print('Current val loss is %.3f' % (np.mean(val_loss_list)))

            val_loss_mean = np.mean(val_loss_list)
            val_reconstruction_loss_mean = np.mean(val_reconstruction_loss)

            writer.add_scalar("val_loss", val_loss_mean, start_epoch)
            writer.add_scalar("mIoU_val_mean", val_miou, start_epoch)
            writer.add_scalar("iou_val_mean", acc_completion, start_epoch)
            writer.add_scalar("val_reconstruction_loss_mean", val_reconstruction_loss_mean, start_epoch)

            my_model.train()

        pbar.close()

        train_loss_mse_mean = np.mean(train_loss_mse)

        for g in optimizer.param_groups:
            lr = g["lr"]
            break
        writer.add_scalar("train_loss_mse_mean", train_loss_mse_mean, start_epoch)
        writer.add_scalar("lr_rate", lr, start_epoch)
        start_epoch += 1

    writer.close()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='/home/Software/AAAI_Code/3D_voxel_diffusion_plane/config/semantickitti_autoencoder.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)

#
