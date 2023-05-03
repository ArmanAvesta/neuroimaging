"""
nnCapsNet Project
Developed by Arman Avesta, MD
Aneja Lab | Yale School of Medicine
Created (11/1/2022)
Updated (11/15/2022)

This file contains tools that help in image visualization, processing, training, and evaluation..
"""

# ------------------------------------------------- ENVIRONMENT SETUP -------------------------------------------------

# Project imports:
from os_tools import list_filter_files_paths

# System imports:
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import os
from os.path import join
from collections import OrderedDict
from typing import Union, Tuple, List

from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage import map_coordinates
from skimage.transform import resize

# Global variables:
ANISO_THRESHOLD = 3

# Print configs:
np.set_printoptions(precision=1, suppress=True)
torch.set_printoptions(precision=1, sci_mode=False)


# ------------------------------------------------ Image Reorientation ------------------------------------------------

def reorient_nifti(nifti):
    """
    Re-orients NIfTI to LAS+ system = standard radiology system.
    Note that the affine transform of NIfTI file (from the MRI volume space to the scanner space) is also corrected.

    :param nifti: input NIfTI file.
    :return: re-oriented NIfTI in LAS+ system.

    Notes:
    ------
    nib.io_orientation compares the orientation of nifti with RAS+ system. So if nifti is already in
    RAS+ system, the return from nib.io_orientation(nifti.affine) will be:
    [[0, 1],
     [1, 1],
     [2, 1]]
    If nifti is in LAS+ system, the return would be:
    [[0, -1],           # -1 means that the first axis is flipped compared to RAS+ system.
     [1, 1],
     [2, 1]]
    If nifti is in PIL+ system, the return would be:
    [[1, -1],           # P is the 2nd axis in RAS+ hence 1 (not 0), and is also flipped hence -1.
     [2, -1],           # I is the 3rd axis in RAS+ hence 2, and is also flipped hence -1.
     [0, -1]]           # L is the 1st axis in RAS+ hence 0, and is also flipped hence -1.
    Because we want to save images in LAS+ orientation rather than RAS+, in the code below we find axis 0 and
    negate the 2nd column, hence going from RAS+ to LAS+. For instance, for PIL+, the orientation will be:
    [[1, -1],
     [2, -1],
     [0, -1]]
    This is PIL+ compared to RAS+. To compare it to LAS+, we should change it to:
    [[1, -1],
     [2, -1],
     [0, 1]]
    That is what this part of the code does:
    orientation[orientation[:, 0] == 0, 1] = - orientation[orientation[:, 0] == 0, 1]
    Another inefficient way of implementing this function is:
    ################################################################################
    original_orientation = nib.io_orientation(nifti.affine)
    target_orientation = nib.axcodes2ornt(('L', 'A', 'S'))
    orientation_transform = nib.ornt_transform(original_orientation, target_orientation)
    return nifti.as_reoriented(orientation_transform)
    ################################################################################
    """
    orientation = nib.io_orientation(nifti.affine)
    orientation[orientation[:, 0] == 0, 1] = - orientation[orientation[:, 0] == 0, 1]
    return nifti.as_reoriented(orientation)


def reorient_niftis(niftis_list, current_suffix='.nii.gz', replaced_suffix='_reoriented.nii.gz'):
    """
    Re-orients a list of NIfTIs to LAS+ system and saves them as new NIfTI files.

    :param niftis_list: List of .nii.gz files, e.g. ['path1.nii.gz', 'path2.nii.gz', 'path3.nii.gz'].
    :param current_suffix: {str} removed and replaced by replaced_suffix.
        e.g. if current_suffix is '.nii.gz' and replaced suffix is '_reoriented.nii.gz', the reoriented files
        are saved with changed paths '_reoriented.nii.gz'
    :param replaced_suffix: {str}
    :return: None.
        Side effect: saves re-oriented NIfTI files in the same folders.
    """
    for nifti_path in tqdm(niftis_list, desc='re-orienting NIfTIs to LAS+'):
        nifti = nib.load(nifti_path)
        orientation = nib.io_orientation(nifti.affine)
        orientation[orientation[:, 0] == 0, 1] = - orientation[orientation[:, 0] == 0, 1]
        nifti = nifti.as_reoriented(orientation)
        nifti_path = nifti_path.replace(current_suffix, replaced_suffix)
        nib.save(nifti, nifti_path)


def reoirent_img(img, voxsize=(1, 1, 1), coords=('L', 'A', 'S')):
    """
    Re-orients a 3D ndarray volume into the standard radiology coordinate system = LAS+.

    :param img: {ndarray}
    :param voxsize: {tuple, list, or ndarray}
    :param coords: {tuple or list}: coordinate system of the image. e.g. ('L','P','S').
        for more info, refer to: http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm

    :return: reoriented image and voxel size in standard radiology coordinate system ('L','A','S) = LAS+ system.
    """
    assert img.ndim == 3, 'image should have shape (x, y, z)'
    if coords == ('L', 'A', 'S'):
        return img, voxsize
    if coords == ('R', 'A', 'S'):
        img = np.flip(img, 0)
        return img, voxsize
    if coords == ('R', 'P', 'I'):
        # R,P,I --> L,A,S
        img = np.flip(img)  # flip the image in all 3 dimensions
        return img, voxsize
    if coords == ('P', 'I', 'R'):
        # P,I,R --> A,S,L
        img = np.flip(img)
        # A,S,L --> L,A,S
        img = np.moveaxis(img, [0, 1, 2], [1, 2, 0])
        voxsize = (voxsize[1], voxsize[2], voxsize[0])
        return img, voxsize
    if coords == ('R', 'I', 'A'):
        # R,I,A --> R,A,I:
        img = np.swapaxes(img, 1, 2)
        voxsize = (voxsize[0], voxsize[2], voxsize[1])
        # R,A,I --> L,A,S:
        img = np.flip(img, [0, 2])
        return img, voxsize
    if coords == ('L', 'I', 'P'):
        # L,I,P --> L,P,I
        img = np.swapaxes(img, 1, 2)
        voxsize = (voxsize[0], voxsize[2], voxsize[1])
        # L,P,I --> L,A,S
        img = np.flip(img, [1, 2])
        return img, voxsize
    if coords == ('P', 'R', 'S'):
        # P,R,S --> R,P,S:
        img = np.swapaxes(img, 0, 1)
        voxsize = (voxsize[1], voxsize[0], voxsize[2])
        # R,P,S --> L,A,S
        img = np.flip(img, [0, 1])
        return img, voxsize
    if coords == ('L', 'I', 'A'):
        img = np.swapaxes(img, 1, 2)
        img = np.flip(img, 2)
        voxsize = (voxsize[0], voxsize[2], voxsize[1])
        return img, voxsize
    if coords == ('L', 'S', 'A'):
        img = np.swapaxes(img, 1, 2)
        voxsize = (voxsize[0], voxsize[2], voxsize[1])
        return img, voxsize
    if coords == ('L', 'P', 'S'):
        img = np.flip(img, 1)
        return img, voxsize
    raise Exception('coords not identified: please revise the reorient_img function and define the coordinate system')


# ----------------------------------------------- Image visualization ------------------------------------------------

def imgshow(img, voxsize=(1, 1, 1), coords=('L', 'A', 'S')):
    """
    This function shows 2D/3D/4D/5D images & image batches:
    - 2D: image is shown.
    - 3D: the volume mid-slices in axial, coronal and sagittal planes are shown.
    - 4D: assumes that image is multichannel image (channel-first) and shows all channels as 3D images.
    - 5D: assumes that image is a batch of multichannel images (batch first, channel second), and shows all
        batches & all channels of 3D images.

    :param img: {ndarray, tensor, or nifti}
    :param voxsize: {tuple, list, or ndarray} Default=(1,1,1)
    :param coords: image coordinate system; Default=('L','A','S') which is the standard radiology coordinate system.
        for more info, refer to: http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm

    :return: None. Side effect: shows image(s).
    """
    if type(img) is nib.nifti1.Nifti1Image:
        voxsize = img.header.get_zooms()
        coords = nib.aff2axcodes(img.affine)
        img = img.get_fdata()
    elif type(img) is torch.Tensor:
        img = img.numpy()

    kwargs = dict(cmap='gray', origin='lower')
    ndim = img.ndim
    assert ndim in (2, 3, 4, 5), f'image shape: {img.shape}; imshow can only show 2D and 3D images, ' \
                                 f'multi-channel 3D images (4D), and batches of multi-channel 3D images (5D).'

    if ndim == 2:
        plt.imshow(img.T, **kwargs)
        plt.show()

    elif ndim == 3:
        img, voxsize = reoirent_img(img, voxsize, coords)
        midaxial = img.shape[2] // 2
        midcoronal = img.shape[1] // 2
        midsagittal = img.shape[0] // 2
        axial_aspect_ratio = voxsize[1] / voxsize[0]
        coronal_aspect_ratio = voxsize[2] / voxsize[0]
        sagittal_aspect_ratio = voxsize[2] / voxsize[1]

        axial = plt.subplot(1, 3, 1)
        plt.imshow(img[:, :, midaxial].T, **kwargs)
        axial.set_aspect(axial_aspect_ratio)
        axial.set_title('axial')

        coronal = plt.subplot(1, 3, 2)
        plt.imshow(img[:, midcoronal, :].T, **kwargs)
        coronal.set_aspect(coronal_aspect_ratio)
        coronal.set_title('coronal')

        sagittal = plt.subplot(1, 3, 3)
        plt.imshow(img[midsagittal, :, :].T, **kwargs)
        sagittal.set_aspect(sagittal_aspect_ratio)
        sagittal.set_title('sagittal')

        plt.show()

    elif ndim > 3:
        for i in range(img.shape[0]):
            imgshow(img[i, ...], voxsize, coords)


# -------------------------------------------------- NIfTI cropping ---------------------------------------------------

def crop_nifti(nifti, msk, pad=1, bg=0):
    """
    This function crops a NIfTI according to the mask.
    :param nifti: nifti image
    :param msk: nifti mask
    :param pad: (int) number of voxels to pad around the cropped image, default=1.
    :param bg: (int) background value, default=0.
    :return:
    """
    msk = np.where(msk != bg)
    minx, maxx = int(np.min(msk[0])) - pad, int(np.max(msk[0])) + pad + 1
    miny, maxy = int(np.min(msk[1])) - pad, int(np.max(msk[1])) + pad + 1
    minz, maxz = int(np.min(msk[2])) - pad, int(np.max(msk[2])) + pad + 1
    '''
    # The longer way of implementing this:
    new_origin_img_space = np.array([minx, miny, minz])
    new_origin_scanner_space = nib.affines.apply_affine(nifti.affine, new_origin_img_space)
    new_affine = nifti.affine.copy()
    new_affine[:3, 3] = new_origin_scanner_space
    img = nifti.get_fdata()
    img_cropped = img[minx:maxx, miny:maxy, minz:maxz]
    nifti_cropped = nib.Nifti1Image(img_cropped, new_affine)
    return nifti_cropped
    '''
    return nifti.slicer[minx:maxx, miny:maxy, minz:maxz]


# ------------------------------------------ Segmentation Labels Pre-Processing ---------------------------------------

def remap_seg_labels(seg_list, remap_dict, current_suffix='.nii.gz', replaced_suffix='.nii.gz'):
    """
    Remap segmentation labels.

    :param seg_list: list of segmentation paths, e.g. ['path1.nii.gz', 'path2.nii.gz', 'path3.nii.gz']
    :param remap_dict: labels are changed from keys to values. So this remap dictionary:
        {0:0, 1:1, 2:2, 3:2, 4:2} will merge 2, 3, and 4 all into 2.
    :param current_suffix: {str}
    :param replaced_suffix: {str}
    :return: None.
        Side effect: saves remapped segmentations in the same folder with added suffix to their name.
    """
    # seg_list = list_and_filter_files_paths('/Users/sa936/projects/sccapsnet/data/images', 'seg.nii.gz')
    # remap_dict = {0:0, 1:2, 2:1, 3:2, 4:2}

    for seg_path in tqdm(seg_list, desc='remapping segmentation labels'):
        seg = nib.load(seg_path)
        seg_data = seg.get_fdata()
        remapped_data = np.zeros_like(seg_data)
        for old_label, new_label in remap_dict.items():
            remapped_data[seg_data == old_label] = new_label
        remapped = nib.Nifti1Image(remapped_data, affine=seg.affine)
        remapped_path = seg_path.replace(current_suffix, replaced_suffix)
        nib.save(remapped, remapped_path)


def calculate_seg_label_percentages(seg_list):
    """
    Calculates the percentage of segmentation label volumes in all images passed in seg_list.

    :param seg_list: list of segmentation paths, e.g. ['path1.nii.gz', 'path2.nii.gz', ...]
    :return: dictionary of segmentation label percentages, e.g. {0: 0.989, 1: 0.001, 2: 0.006, 4: 0.001}
    """
    # seg_list = list_and_filter_files_paths('/Users/sa936/projects/sccapsnet/data/images', 'seg.nii.gz')

    unique_labels = set()
    for seg_path in tqdm(seg_list, desc='extracting unique labels'):
        seg_data = nib.load(seg_path).get_fdata()
        unique_labels = unique_labels | set(np.unique(seg_data))

    unique_labels_counts = {key: 0 for key in unique_labels}

    for seg_path in tqdm(seg_list, desc='computing segmentation label counts'):
        seg_data = nib.load(seg_path).get_fdata()
        labels, counts = np.unique(seg_data, return_counts=True)
        for i, label in enumerate(labels):
            unique_labels_counts[label] += counts[i]

    s = sum(unique_labels_counts.values())
    unique_label_percentages = {label: count / s for label, count in unique_labels_counts.items()}

    return unique_label_percentages


# ------------------------------------------------ Image Resampling ---------------------------------------------------

def get_do_separate_z(spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      anisotropy_threshold=ANISO_THRESHOLD):
    """
    Returns True if we should use a separate interpolation method for the high-voxel-spacing axis (here arbitrarily
    named z axis) because the voxel spacing along that that axis is way more than other axes.

    :param spacing: {tuple, list, or ndarray} voxel spacing, e.g. (.9, .9, 3)
    :param anisotropy_threshold: anisotropy threshold for high-spacing voxel axis
    :return: {bool} True if we should use a separate interpolation method for high-voxel-spacing axis.
    """
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(voxel_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    """
    Finds which axis (x, y, or z) has high voxel spacing.

    :param voxel_spacing: {tuple, list, or ndarray} e.g. (0.7, 0.7, 3)
    :return: {int or None} the axis with highest pixel spacing, e.g. 2. If two or more axes have similar large
        spacing --> function returns None.

    Notes:
    ------
    If voxel_spacing is (0.24, 1.25, 1.25), the function returns None because more than one  axis has equally-large
    voxel spacing.
    Beware that we cannot use voxel_spacing.argmax() in our code because it only returns one largest and omits
    multiple maxima. That's why we use np.where to find both more than one maxima.

    If voxel_spacing is (1.25, 1.25, 1.25), the function returns None.

    Explaining the indexing here:
        voxel_spacing = np.array((.7, .7, 3))
        axis = np.where(voxel_spacing == max(voxel_spacing)) --> axis = (array([2]),)
        axis = np.where(voxel_spacing == max(voxel_spacing))[0] --> axis = array([2])
        axis = np.where(voxel_spacing == max(voxel_spacing))[0][0] --> axis = 2
    """
    voxel_spacing = np.array(voxel_spacing)
    axis = np.where(voxel_spacing == max(voxel_spacing))[0]  # find which axes have highest spacing.
    return axis[0] if len(axis) <= 1 else None


def compute_new_shape(old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                      old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    """
    Calculates the new shape of image given old and new voxel spacings.

    :param old_shape: e.g. (128, 128, 64)
    :param old_spacing: e.g. (1, 1, 2)
    :param new_spacing: e.g. (1, 1, 1)
    :return: new shape of image, e.g. (128, 128, 128)
    """
    print(f'old spacing: {old_spacing}, old shape: {old_shape}, new spacing: {new_spacing}')
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape


def resample_data_or_seg_to_spacing(data: np.ndarray,
                                    orig_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                    is_seg: bool = False,
                                    order: int = 3,
                                    order_z: int = 0,
                                    separate_z_anisotropy_threshold=ANISO_THRESHOLD):
    """
    Resamples image or segmentation.

    :param data: image or segmentation; shape: [c, x, y, z]; c: channels
    :param orig_spacing: [a, b, c]
    :param new_spacing: [d, e, f]
    :param is_seg: if True, data is segmentation; if False, data is a scalar image.
    :param order: interpolation order.
    :param order_z: interpolation order in the high-spacing direction.
    :param separate_z_anisotropy_threshold: e.g. if threshold is 3 and voxel spacing is [.5, .5, 3],
        separate interpolation method is used along z axis.
    :return:
    """
    assert data.ndim == 4, 'data must have the shape [c, x, y, z]'
    axis = None

    if get_do_separate_z(orig_spacing, separate_z_anisotropy_threshold):
        axis = get_lowres_axis(orig_spacing)
    elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
        axis = get_lowres_axis(new_spacing)

    orig_shape = np.array(data[0].shape)     # [x, y, z] without channels
    new_shape = compute_new_shape(orig_shape, orig_spacing, new_spacing)        # [xn, yn, zn] without channels

    data_resampled = resample_data_or_seg(data, new_shape, is_seg, axis, order, order_z=order_z)
    return data_resampled       # [c, xn, yn, zn]


def resample_data_or_seg_to_shape(data: np.ndarray,
                                  new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                                  orig_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  is_seg: bool = False,
                                  order: int = 3,
                                  order_z: int = 0,
                                  force_separate_z: Union[bool, None] = False,
                                  anisotropy_threshold=ANISO_THRESHOLD):
    """
    needed for segmentation export. Stupid, I know. Maybe we can fix that with Leos new resampling functions
    """
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(orig_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(orig_spacing, anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(orig_spacing)
        elif get_do_separate_z(new_spacing, anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the same spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        assert len(data.shape) == 4, "data must be c x y z"

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, order_z=order_z)
    return data_reshaped


def resample_data_or_seg(data: np.ndarray,
                         new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
                         is_seg: bool = False,
                         axis: Union[None, int] = None,
                         order: int = 3,
                         order_z: int = 0):
    """

    :param data: image or segmentation data
    :param new_shape: e.g. [x, y, z]
    :param is_seg: True or False
    :param axis: if None: don't use separate interpolation method along high-voxel-spacing axis;
        if an integer number: use separate interpolation method along the axis.
    :param order: interpolation order.
    :param order_z: interpolation order along high-voxel-spacing axis; only used if axis is not None.
    :return: resampled image or segmentation
    """
    assert data.ndim == 4, "data must have the shape [c, x, y, z]"
    assert len(new_shape) == data.ndim - 1, "new_shape must have the shape [x_new, y_new, z_new]"

    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}

    data_dtype = data.dtype
    c, x, y, z = data.shape
    xn, yn, zn = new_shape
    orig_shape = np.array(data[0].shape)  # [x, y, z]
    new_shape = np.array(new_shape)  # [xn, yn, zn]

    if (orig_shape == new_shape).all():
        # print("no resampling necessary")
        return data

    data = data.astype(float)

    if axis is None:
        # print("no separate z, order", order)
        resampled_data = np.zeros([c, xn, yn, zn], dtype=data_dtype)
        for channel in range(c):
            resampled_data[channel, ...] = resize_fn(data[channel, ...], new_shape, order, **kwargs).astype(data_dtype)
        return resampled_data

    if axis == 0:
        new_shape_2d = new_shape[1:]
    elif axis == 1:
        new_shape_2d = new_shape[[0, 2]]
    elif axis == 2:
        new_shape_2d = new_shape[:-1]
    else:
        raise Exception('axis should be one of None, 1, 2, or 3')

    resampled_data = []

    for c in range(data.shape[0]):  # c: channel dimension
        '''
        make resampled channels separately and then combine them back into multi-channel image
        because we don't want to interpolate between channels.
        for each channel: resample [x, y, z] --> [xn, yn, zn]
        '''
        resampled_channel = []  # channel is composed of slices

        for slice_id in range(orig_shape[axis]):
            '''
            for each slice: resample [x, y] --> [xn, yn]
            '''
            if axis == 0:
                resampled_channel.append(resize_fn(data[c, slice_id, :, :], new_shape_2d, order, **kwargs))
            elif axis == 1:
                resampled_channel.append(resize_fn(data[c, :, slice_id, :], new_shape_2d, order, **kwargs))
            else:
                resampled_channel.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs))

        resampled_channel = np.stack(resampled_channel, axis)

        if orig_shape[axis] == new_shape[axis]:
            '''
            if z == zn: no inter-slice interpolation is needed because the resampled image has the same number of 
            slices as the original image.
            '''
            resampled_data.append(resampled_channel[None])

        else:
            '''
            here z != zn --> we should interpolate between slices because resampled image has different number of 
            slices compared to original image. Thus, we should go from resampled slices [xn, yn, z] to resampled image
            [xn, yn, zn]. 
            '''
            rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
            orig_rows, orig_cols, orig_dim = resampled_channel.shape

            row_scale = float(orig_rows) / rows
            col_scale = float(orig_cols) / cols
            dim_scale = float(orig_dim) / dim

            map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
            map_rows = row_scale * (map_rows + 0.5) - 0.5
            map_cols = col_scale * (map_cols + 0.5) - 0.5
            map_dims = dim_scale * (map_dims + 0.5) - 0.5

            coord_map = np.array([map_rows, map_cols, map_dims])

            if not is_seg or order_z == 0:
                resampled_data.append(map_coordinates(resampled_channel, coord_map,
                                                      order=order_z, mode='nearest')[None])
            else:
                unique_labels = np.sort(pd.unique(resampled_channel.ravel()))  # np.unique(resampled_channel)
                reshaped = np.zeros(new_shape, dtype=data_dtype)

                for i, cl in enumerate(unique_labels):
                    reshaped_multihot = np.round(
                        map_coordinates((resampled_channel == cl).astype(float), coord_map, order=order_z,
                                        mode='nearest'))
                    reshaped[reshaped_multihot > 0.5] = cl
                resampled_data.append(reshaped[None])

    resampled_data = np.vstack(resampled_data)
    return resampled_data



# -------------------------------------------------- CODE TESTING -----------------------------------------------------

if __name__ == '__main__':
    print(f'''
    
    ''')
    img = nib.load('/Users/arman/projects/sccapsnet/data/ucsf/UCSF-PDGM-0004_nifti/t1c.nii.gz')
    imgshow(img)


