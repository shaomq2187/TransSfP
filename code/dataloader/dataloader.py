# -*-coding:utf-8-*-
import os
import glob
import sys
import time
import numpy
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia
import imageio
import API.utils
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# from API.utils import exr_loader
class TransSfPDataset(Dataset):
    """
    Dataset class for testing model on estimation of surface normals.
    Uses imgaug for image augmentations.
    If a label_dir is blank ( None, ''), it will assume labels do not exist and return a tensor of zeros
    for the label.
    Args:
        input_I_sum_dir (str): Path to folder containing the I_sum imgs(.png format)
        input_normal_dir (str): Root dir  to folder containing the four input normal dirs (synthesis-normal-0,-1,-2,-3) (.exr format)
        label_dir (str): (Optional) Path to folder containing the labels (.exr format).
                         If no labels exists, pass empty string ('') or None.
        mask_dir (str): (Optional) Path to folder containing the masks (.png format).
                         If no labels exists, pass empty string ('') or None.
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs
        input_only (list, str): List of transforms that are to be applied only to the input img
    """

    def __init__(self,  label_dir,dolp_dir,aolp_dir,synthesis_normals_dir,mask_dir=None, transform=None, input_only=None):
        self.dolp_dir = dolp_dir
        self.aolp_dir = aolp_dir
        self.synthesis_normals_dir = synthesis_normals_dir
        self.labels_dir = label_dir
        self.masks_dir = mask_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_dolp = []
        self._datalist_aolp = []
        self._datalist_input_normal_0 = []
        self._datalist_input_normal_1 = []
        self._datalist_input_normal_2 = []
        self._datalist_input_normal_3 = []
        self._datalist_label = []
        self._datalist_mask = []
        self._create_lists_filenames()

    def __len__(self):
        return len(self._datalist_label)

    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index. If no labels directory has been specified,
        then a tensor of zeroes will be returned as the label.
        Args:
            index (int): index of the item required from dataset.
        Returns:
            torch.Tensor: Tensor of input image, shape is (13, Height, Width), the 13 channels are (norm_0 x 3 , norm_1 x 3, norm_2 x 3, norm_3 x 3, intensity)
            torch.Tensor: Tensor of label (Tensor of zeroes is labels_dir is "" or None), the shape (3, Height, Width) for real norm
            torch.Tensor: Tensor of mask ( The shape (3, 1, Height, Width) for real norm
        '''

        # -- 1、prepare paths
        # all_time = time.time()
        dolpPath = self._datalist_dolp[index]
        aolpPath = self._datalist_aolp[index]
        norm_0_path = self._datalist_input_normal_0[index]
        norm_1_path = self._datalist_input_normal_1[index]
        norm_2_path = self._datalist_input_normal_2[index]
        norm_3_path = self._datalist_input_normal_3[index]


        if(self.masks_dir is not None):
            mask_path = self._datalist_mask[index]
        label_path = self._datalist_label[index]
        # start = time.time()
        # print(index)
        # print(I_sum_path)
        # print(norm_0_path)
        # print(norm_1_path)
        # print(norm_2_path)
        # print(norm_3_path)
        # print(mask_path)
        # print(label_path)
        # print('')
        # -- 2、load imgs
        dolpImg = API.utils.rgb_loader(dolpPath)  # numpy array shape is (height, width, 3)
        aolpImg = API.utils.rgb_loader(aolpPath)
        norm_0 = API.utils.rgb_loader(norm_0_path)  # numpy array shape is (height, width, 3)
        norm_1 = API.utils.rgb_loader(norm_1_path)
        norm_2 = API.utils.rgb_loader(norm_2_path)
        norm_3 = API.utils.rgb_loader(norm_3_path)
        mask_img = API.utils.mask_loader(mask_path)
        label_img = API.utils.rgb_loader(label_path).transpose(2, 0, 1) # To( C,H,W)
        # print("load time",time.time()-start)

        # -- 3、concat inputs
        # start = time.time()
        height = label_img.shape[1]
        width = label_img.shape[2]
        input_img_arr = numpy.zeros([14, height, width], dtype=np.uint8)  # shape is (13 x H x W)
        input_img_arr[0,:,:] = dolpImg
        input_img_arr[1,:,:] = aolpImg
        input_img_arr[2:5, :, :] = norm_0.transpose(2, 0, 1)  # 3 x H x W
        input_img_arr[5:8, :, :] = norm_1.transpose(2, 0, 1)
        input_img_arr[8:11, :, :] = norm_2.transpose(2, 0, 1)
        input_img_arr[11:14, :, :] = norm_3.transpose(2, 0, 1)


        # print("concat time",time.time()-start)

        # -- 5、apply mask to inputs and label
        if(self.masks_dir is not None):
            # start = time.time()
            input_img_arr[:, mask_img == 0] = 0
            label_img[:, mask_img == 0] = 0

        # print("apply mask time",time.time()-start)

        # -- 4、apply image augmentations
        # start = time.time()

        if self.transform:
            # apply augment to inputs
            det_tf = self.transform.to_deterministic()
            input_img_arr = det_tf.augment_image(
                input_img_arr.transpose(1, 2, 0))  # augment_image require shape (H, W, C)
            input_img_arr = input_img_arr.transpose(2, 0, 1)  # To Shape: (13, H, W)

            # apply augment to label
            label_img = det_tf.augment_image(label_img.transpose(1, 2, 0), hooks=ia.HooksImages(
                activator=self._activator_masks))  # some transforms only appy to inputs, not label
            label_img = label_img.transpose(2, 0, 1)  # To (C,H,W)
            # apply mask
            if(self.masks_dir is not None):
                mask_img = det_tf.augment_image(mask_img, hooks=ia.HooksImages(
                    activator=self._activator_masks))  # some transforms only appy to inputs, not label

        # -- 4、normalize
        # start = time.time()

        input_tensor = transforms.ToTensor()(
            input_img_arr.copy().transpose(1, 2, 0))  # ToTensor contains the normalization process

        params_tensor = input_tensor[0:2,:,:]
        synthesis_normals_tensor = input_tensor[2:14,:,:]
        # convert 0-255 to (-1,1)
        label_tensor = torch.from_numpy(label_img).float()  # (C,H,W)
        label_tensor = (label_tensor-127)/127.0
        label_tensor = nn.functional.normalize(label_tensor,p=2,dim=0)
        if(self.masks_dir is not None):
            mask_tensor = torch.from_numpy(mask_img.copy()).unsqueeze(0)


        if(self.masks_dir is not None):
            return params_tensor,synthesis_normals_tensor, label_tensor, mask_tensor
        else:
            return params_tensor,synthesis_normals_tensor, label_tensor


    def _create_lists_filenames(self):
        '''Creates a list of filenames of images and labels each in dataset
        The label at index N will match the image at index N.
        Args: None
        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and labels do not match
        '''
        assert os.path.isdir(self.dolp_dir), 'Dataloader given input DoLP directory that does not exist: "%s"' % (self.dolp_dir)
        assert os.path.isdir(self.aolp_dir), 'Dataloader given input AoLP directory that does not exist: "%s"' % (self.aolp_dir)
        assert os.path.isdir(self.synthesis_normals_dir),'Dataloader given input synthesis normals directory that does not exist: "%s"' % (self.synthesis_normals_dir)
        assert os.path.isdir(self.labels_dir), 'Dataloader given labels directory that does not exist: "%s"' % (self.labels_dir)
        if(self.masks_dir is not None):
            assert os.path.isdir(
                self.masks_dir), 'Dataloader given masks_dir images directory that does not exist: "%s"' % (self.masks_dir)




        # -- input DoLP & AoLP images
        dolp_search_str = os.path.join(self.dolp_dir, '*.png')
        self._datalist_dolp = sorted(glob.glob(dolp_search_str))
        numDoLP = len(self._datalist_dolp)
        if numDoLP == 0:
            raise ValueError('No input DoLP files found in given directory. Searched in dir: {} '.format(
                dolp_search_str))
        aolp_search_str = os.path.join(self.aolp_dir, '*.png')
        self._datalist_aolp = sorted(glob.glob(aolp_search_str))
        numNAoLP = len(self._datalist_aolp)
        if numNAoLP == 0:
            raise ValueError('No input AoLP files found in given directory. Searched in dir: {} '.format(
                aolp_search_str))

        # -- input normal images
        input_normal_0_search_str = os.path.join(self.synthesis_normals_dir, 'synthesis-normal-0')
        input_normal_0_search_str = os.path.join(input_normal_0_search_str, '*-normal.png')
        self._datalist_input_normal_0 = sorted(glob.glob(input_normal_0_search_str))
        numNorm_0 = len(self._datalist_input_normal_0)
        if numNorm_0 == 0:
            raise ValueError('No input normal_0 files found in given directory. Searched in dir: {} '.format(
                input_normal_0_search_str))

        input_normal_1_search_str = os.path.join(self.synthesis_normals_dir, 'synthesis-normal-1')
        input_normal_1_search_str = os.path.join(input_normal_1_search_str, '*-normal.png')
        self._datalist_input_normal_1 = sorted(glob.glob(input_normal_1_search_str))
        numNorm_1 = len(self._datalist_input_normal_1)
        if numNorm_1 == 0:
            raise ValueError('No input normal_1 files found in given directory. Searched in dir: {} '.format(
                input_normal_1_search_str))

        input_normal_2_search_str = os.path.join(self.synthesis_normals_dir, 'synthesis-normal-2')
        input_normal_2_search_str = os.path.join(input_normal_2_search_str, '*-normal.png')
        self._datalist_input_normal_2 = sorted(glob.glob(input_normal_2_search_str))
        numNorm_2 = len(self._datalist_input_normal_2)
        if numNorm_2 == 0:
            raise ValueError('No input normal_2 files found in given directory. Searched in dir: {} '.format(
                input_normal_2_search_str))

        input_normal_3_search_str = os.path.join(self.synthesis_normals_dir, 'synthesis-normal-3')
        input_normal_3_search_str = os.path.join(input_normal_3_search_str, '*-normal.png')
        self._datalist_input_normal_3 = sorted(glob.glob(input_normal_3_search_str))
        numNorm_3 = len(self._datalist_input_normal_3)
        if numNorm_3 == 0:
            raise ValueError('No input normal_3 files found in given directory. Searched in dir: {} '.format(
                input_normal_3_search_str))
        if not (numNorm_0 == numNorm_1 == numNorm_2 == numNorm_3):
            raise ValueError('Numbers of input normals are different.')


        # -- 3、labels(real normals)
        labels_search_str = os.path.join(self.labels_dir, '*.png')
        self._datalist_label = sorted(glob.glob(labels_search_str))
        numLabels = len(self._datalist_label)
        if numLabels == 0:
            raise ValueError(
                'No input label files found in given directory. Searched in dir: {} '.format(self.labels_dir))

        # -- 4、masks
        if(self.masks_dir is not None):
            masks_search_str = os.path.join(self.masks_dir, '*.png')
            self._datalist_mask = sorted(glob.glob(masks_search_str))
            numMasks = len(self._datalist_mask)
            if numMasks == 0:
                raise ValueError(
                    'No input mask files found in given directory. Searched in dir: {} '.format(self.masks_dir))

        # -- 5、verify number of every input
        if not ( numNorm_0 == numLabels == numLabels):
            print(numNorm_0, numNorm_1, numNorm_2, numNorm_3, numMasks, numLabels)
            raise ValueError('Numbers of inputs(rgb,normal,label,mask) are different.')


    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default


if (__name__ == '__main__'):
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision

    # Example Augmentations using imgaug
    imsize = 512
    augs_train = iaa.Sequential([
        # Geometric Augs
        iaa.Resize((imsize, imsize)),  # Resize image
        # iaa.Fliplr(0.5),
        # iaa.Flipud(0.5),
        # iaa.Rot90((0, 4)),

        # Blur and Noise
        # iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
        # iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
        # iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),

        # Color, Contrast, etc.
        # iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
        # iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
        # iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
        # iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
    ])
    augs_test = iaa.Sequential([
        # Geometric Augs
        iaa.Resize((imsize, imsize), 0),
    ])

    augs = augs_train
    input_only = ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]

    dt_train = RealSurfaceNormalsDataset(
        input_I_sum_dir='/media/zjw/data/smq/samples/Middle-Round-Cup-2/PolarImg-I-sum/8-Bit',
        input_normal_dir='/media/zjw/data/smq/samples/Middle-Round-Cup-2/synthesis-normals',
        label_dir='/media/zjw/data/smq/samples/Middle-Round-Cup-2/Normals-PNG',
        mask_dir='/media/zjw/data/smq/samples/Middle-Round-Cup-2/Masks')

    # print("dataset")
    # batch_size = 16
    # testloader = DataLoader(dt_train, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True,prefetch_factor=2)
    # print("dataloader")
    # # Show 1 Shuffled Batch of Images
    # import loss_functions
    # for ii, batch in enumerate(testloader):
    #     # Get Batch
    #     input_tensor, label_tensor,mask_tensor = batch
    #     print("ii:",ii)
    #
    #     print(" ")
    #     print(" ")
    #     print("input_vec:", input_tensor[4:7, :, :].shape)
    #     print("target_vec:", label_tensor.shape)
    #     print("mask_vec:", mask_tensor.squeeze(1).shape)
    #     print('loss', loss_functions.loss_fn_cosine(input_vec=input_tensor[:,10:13, :, :],
    #                                                 target_vec=label_tensor,
    #                                                 mask_tensor=mask_tensor.squeeze(1),
    #                                                 reduction='elementwise_mean'))
    #     loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
    #         input_tensor[:,4:7, :, :], label_tensor.double(), mask_tensor.squeeze(1))
    #     print("loss_deg_mean:",loss_deg_mean)
    #     print("loss_deg_median:",loss_deg_median)
    #     print("percentage_1:",percentage_1)
    #     print("percentage_2:",percentage_2)
    #     print("percentage_3:",percentage_3)
    #
    #     # # Show Batch
    #     # sample = torch.cat((img[1:,:,:], label), 5)
    #     # im_vis = torchvision.utils.make_grid(sample, nrow=batch_size // 4, padding=2, normalize=True, scale_each=True)
    #     # plt.imshow(im_vis.numpy().transpose(1, 2, 0))
    #     # plt.show()
    #
    #     # break

    import loss_functions

    input_tensor, label_tensor, mask_tensor = dt_train.__getitem__(11)
    input_img_arr = input_tensor.numpy()
    label_img = label_tensor.numpy()
    mask_img = mask_tensor.numpy()
    fig = plt.figure()
    ax0 = plt.subplot(241)
    ax0.imshow(label_img.transpose(1, 2, 0))
    ax1 = plt.subplot(245)
    ax1.imshow(input_img_arr[1:4, :, :].transpose(1, 2, 0))
    ax2 = plt.subplot(246)
    ax2.imshow(input_img_arr[4:7, :, :].transpose(1, 2, 0))
    ax3 = plt.subplot(247)
    ax3.imshow(input_img_arr[7:10, :, :].transpose(1, 2, 0))
    ax4 = plt.subplot(248)
    ax4.imshow(input_img_arr[10:13, :, :].transpose(1, 2, 0))
    ax5 = plt.subplot(242)
    ax5.imshow(input_img_arr[0, :, :])
    ax6 = plt.subplot(243)
    ax6.imshow(mask_img.squeeze(0))
    print("mask_valid:_nums:", len(np.where(mask_img > 0)[1]))
    print(mask_img.shape)
    print("input_vec:", input_tensor[4:7, :, :].unsqueeze(0).shape)
    print("target_vec:", label_tensor.unsqueeze(0).shape)
    print("mask_vec:", mask_tensor.unsqueeze(0).shape)
    print('loss', loss_functions.loss_fn_cosine(input_vec=input_tensor[1:4, :, :].unsqueeze(0),
                                                target_vec=label_tensor.unsqueeze(0),
                                                mask_tensor=mask_tensor.unsqueeze(0).squeeze(1),
                                                reduction='elementwise_mean'))
    loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
        input_tensor[10:13, :, :].unsqueeze(0), label_tensor.double().unsqueeze(0))
    print("loss_deg_mean:", loss_deg_mean)
    print("loss_deg_median:", loss_deg_median)
    print("percentage_1:", percentage_1)
    print("percentage_2:", percentage_2)
    print("percentage_3:", percentage_3)

    plt.show()