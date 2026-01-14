import os
import numpy as np
from torch.utils import data
import yaml
import pathlib

REGISTERED_PC_DATASET_CLASSES = {}

def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]


@register_dataset
class SemKITTI_sk(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="semantic-kitti.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    remove_ind = label == 0
    label -= 1
    label[remove_ind] = 255
    return label


def unpack(compressed):  # from samantickitti api
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed


def get_eval_mask(labels, invalid_voxels):  # from samantickitti api
    """
    Ignore labels set to 255 and invalid voxels (the ones never hit by a laser ray, probed using ray tracing)
    :param labels: input ground truth voxels
    :param invalid_voxels: voxels ignored during evaluation since the lie beyond the scene that was captured by the laser
    :return: boolean mask to subsample the voxels to evaluate
    """
    masks = np.ones_like(labels, dtype=np.bool_)
    masks[labels == 255] = False
    masks[invalid_voxels == 1] = False

    return masks


from os.path import join


@register_dataset
class SemKITTI_sk_multiscan(data.Dataset):
    def __init__(self, data_path, imageset='train', return_ref=False, label_mapping="semantic-kitti-multiscan.yaml",
                 nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        ### remap completion label
        remapdict = semkittiyaml['learning_map']
        # make lookup table for mapping
        maxkey = max(remapdict.keys())
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())
        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        # remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.
        self.comletion_remap_lut = remap_lut

        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.data_path = data_path
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        multiscan = 0  # additional frames are fused with target-frame. Hence, multiscan+1 point clouds in total
        print('multiscan: %d' % multiscan)
        self.multiscan = multiscan
        self.im_idx = []

        self.calibrations = []
        self.times = []
        self.poses = []
        self.point_label = []

        # self.load_calib_poses()
        for i_folder in split:
            # velodyne path corresponding to voxel path
            complete_path = os.path.join(data_path,"sequences", str(i_folder).zfill(2), "voxels")
            files = list(pathlib.Path(complete_path).glob('*.bin'))
            for filename in files:
                self.im_idx.append(str(filename))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        self.times = []
        self.poses = []

        for seq in range(0, 22):
            seq_folder = join(self.data_path, str(seq).zfill(2))

            # Read Calib
            self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def fuse_multi_scan(self, points, pose0, pose):

        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
        new_points = new_points[:, :3]
        new_coords = new_points - pose0[:3, 3]
        new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
        new_coords = np.hstack((new_coords, points[:, 3:]))

        return new_coords

    def __getitem__(self, index):
        ## VAE Encoder Features
        path_VAE = self.im_idx[index]
        raw_data = np.fromfile(path_VAE, dtype=np.float32).reshape((-1, 1))
        ## Image Condition Features
        path_image = path_VAE.replace('VAE_Encoder_Features_One_To_One', 'Image_transform_Voxel_Condition_Features')
        image_data = np.fromfile(path_image, dtype=np.float32).reshape((-1, 1))
        ## Partial Condition Features
        path_partial = path_VAE.replace('VAE_Encoder_Features_One_To_One', 'Condition_Features_2')
        partial_data = np.fromfile(path_partial, dtype=np.float32).reshape((-1, 1))

        ## GroundTruth
        path_GT = path_VAE.replace('/home/SSC_CODE/Data/VAE_Encoder_Features_One_To_One/', '/hdd/Data/data_odometry_voxel_all_with_one/').replace('.bin', '.label')
        GT_data = np.fromfile(path_GT, dtype=np.int16).reshape((-1, 1))
        annotated_data_GT = GT_data
        annotated_data_GT = annotated_data_GT.reshape((256, 256, 32))
        origin_len = len(raw_data)
        number_idx = int(self.im_idx[index][-10:-4])
        dir_idx = int(self.im_idx[index][-20:-18])
        annotated_data_VAE = raw_data.reshape((8, 64, 64, 8))
        annotated_data_image = image_data.reshape((8, 64, 64, 8))
        annotated_data_partial = partial_data.reshape((8, 64, 64, 8))

        data_tuple = (annotated_data_GT.astype(np.uint8),annotated_data_VAE, annotated_data_partial, annotated_data_image, origin_len, dir_idx, number_idx)  # xyz, voxel labels

        return data_tuple


# load Semantic KITTI class info
def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name


def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]

    return nuScenes_label_name
