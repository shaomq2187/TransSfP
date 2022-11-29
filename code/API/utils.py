# -*-coding:utf-8-*-
import os.path

import imageio
import numpy as np
# import Imath
# import OpenEXR
from PIL import Image
# import torch
# from torchvision.utils import make_grid


# def exr_loader(EXR_PATH, ndim=3):
#     """Loads a .exr file as a numpy array
#     Args:
#         EXR_PATH: path to the exr file
#         ndim: number of channels that should be in returned array. Valid values are 1 and 3.
#                         if ndim=1, only the 'R' channel is taken from exr file
#                         if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
#                             The exr file must have 3 channels in this case.
#     Returns:
#         numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
#                                           If ndim=3, shape is (height x width x 3)
#     """
#
#     exr_file = OpenEXR.InputFile(EXR_PATH)
#     cm_dw = exr_file.header()['dataWindow']
#     size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)
#
#     pt = Imath.PixelType(Imath.PixelType.FLOAT)
#
#     if ndim == 3:
#         # read channels indivudally
#         allchannels = []
#         for c in ['R', 'G', 'B']:
#             # transform data to numpy
#             channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
#             channel.shape = (size[1], size[0])
#             allchannels.append(channel)
#
#         # create array and transpose dimensions to match tensor style
#         exr_arr = np.array(allchannels).transpose(( 1, 2,0))
#         return exr_arr
#
#     if ndim == 1:
#         # transform data to numpy
#         channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
#         channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
#         exr_arr = np.array(channel)
#         return exr_arr
def rgb_loader(RGB_PATH):
    """Loads a .jpg file as a numpy array
    Args:
        RGB_PTTH: path to the rgb file
    Returns:
        numpy.ndarray (dtype=np.uint8): shape is (height x width x 3) with (R,G,B) channel   """
    rgb_arr = imageio.imread(RGB_PATH)

    return rgb_arr
def mask_loader(MASK_PATH):
    """Loads segmentation mask file as a numpy array
    Args:
        MASK_PATH: path to the mask file
    Returns:
        numpy.ndarray (dtype=np.uint8): shape is (height x width)   """
    return imageio.imread(MASK_PATH)

def rgb2grey(RGB_ARR):
    """Convert RGB array to a grey numpy array
    Args:
        RGB_ARR: numpy.narray (dtype=np.uint8) of rgb image, shape is (height x width x 3)

    Returns:
        numpy.ndarray (dtype=np.unit8): shape is (height x width)   """

    return np.dot(RGB_ARR[..., :3], [0.3333333333, 0.3333333333, 0.3333333333])

def normal_to_rgb(normals_to_convert):
    '''Converts a surface normals array into an RGB image.
    Surface normals are represented in a range of (-1,1),
    This is converted to a range of (0,255) to be written
    into an image.
    The surface normals are normally in camera co-ords,
    with positive z axis coming out of the page. And the axes are
    mapped as (x,y,z) -> (R,G,B).
    '''
    camera_normal_rgb = normals_to_convert + 1   #transform (-1,1) to (0,2)
    camera_normal_rgb *= 127.5 #(transform(0,2)to(0,255))
    camera_normal_rgb = camera_normal_rgb.astype(np.uint8)
    return camera_normal_rgb
def scaledimg2floatimg(input_arr):
    '''Converts a uint8 array(0,255) into a float32 array(0,1).
    Map (0,255) to the range of (0,1)
    Args:
        input_arr: a uint8 array, its value is range form 0 to 255
    Returns:
        numpy.ndarray (dtype=np.float32): shape is same with input_arr   """
    '''
    output_arr = input_arr/255.0;
    return  output_arr
def png_saver(PNG_PATH,nadrr):
    '''Saves a numpy array as an png file with dtype of np.unit8
    Args:
        PNG_PATH (str): The path to which file will be saved, shape is (H x W x 3)
        ndarr (ndarray): A numpy array containing img data

    Returns:
        None
    '''
    imageio.imwrite(PNG_PATH,nadrr)
def exr_saver(EXR_PATH, ndarr, ndim=3):
    '''Saves a numpy array as an EXR file with HALF precision (float16)
    Args:
        EXR_PATH (str): The path to which file will be saved
        ndarr (ndarray): A numpy array containing img data
        ndim (int): The num of dimensions in the saved exr image, either 3 or 1.
                        If ndim = 3, ndarr should be of shape (height, width) or (3 x height x width),
                        If ndim = 1, ndarr should be of shape (height, width)
    Returns:
        None
    '''
    if ndim == 3:
        # Check params
        if len(ndarr.shape) == 2:
            # If a depth image of shape (height x width) is passed, convert into shape (3 x height x width)
            ndarr = np.stack((ndarr, ndarr, ndarr), axis=0)

        if ndarr.shape[0] != 3 or len(ndarr.shape) != 3:
            raise ValueError(
                'The shape of the tensor should be (3 x height x width) for ndim = 3. Given shape is {}'.format(
                    ndarr.shape))

        # Convert each channel to strings
        Rs = ndarr[0, :, :].astype(np.float16).tostring()
        Gs = ndarr[1, :, :].astype(np.float16).tostring()
        Bs = ndarr[2, :, :].astype(np.float16).tostring()

        # Write the three color channels to the output file
        HEADER = OpenEXR.Header(ndarr.shape[2], ndarr.shape[1])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])

        out = OpenEXR.OutputFile(EXR_PATH, HEADER)
        out.writePixels({'R': Rs, 'G': Gs, 'B': Bs})
        out.close()