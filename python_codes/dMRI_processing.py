# Universal variables & functions
# -----------------------------------------------------------------------------
import numpy as np
# Set numpy to print 3 decimal points and suppress small values
np.set_printoptions(precision=2, suppress=True)

# default voxel size:
voxel_size = (1, 1, 1)


def show_image(img, voxel_size=voxel_size):
    '''
    This function shows the mid-slices of MRI volumes.
    In case the volume is a diffusion image, b0 and mid-diffusion images are shown
    with the correct coronal/sagittal aspect ratios.
    '''
    import matplotlib.pyplot as plt

    dim = len(img.shape)
    if dim in (3, 4):
        midaxial = img.shape[2] // 2
        midcoronal = img.shape[1] // 2
        midsagittal = img.shape[0] // 2
        axial_aspect_ratio = voxel_size[1] / voxel_size[0]
        coronal_aspect_ratio = voxel_size[2] / voxel_size[0]
        sagittal_aspect_ratio = voxel_size[2] / voxel_size[1]

    if dim == 4 and img.shape[3] != 1:
        middiff = img.shape[3] // 2

        axial_b0 = plt.subplot(2, 3, 1)
        plt.imshow(img[:, :, midaxial, 0].T, cmap="gray", origin="lower")
        axial_b0.set_aspect(axial_aspect_ratio)

        coronal_b0 = plt.subplot(2, 3, 2)
        plt.imshow(img[:, midcoronal, :, 0].T, cmap="gray", origin="lower")
        coronal_b0.set_aspect(coronal_aspect_ratio)

        sagittal_b0 = plt.subplot(2, 3, 3)
        plt.imshow(img[midsagittal, :, :, 0].T, cmap="gray", origin="lower")
        sagittal_b0.set_aspect(sagittal_aspect_ratio)

        axial_middiff = plt.subplot(2, 3, 4)
        plt.imshow(img[:, :, midaxial, middiff].T, cmap="gray", origin="lower")
        axial_middiff.set_aspect(axial_aspect_ratio)

        coronal_middiff = plt.subplot(2, 3, 5)
        plt.imshow(img[:, midcoronal, :, middiff].T, cmap="gray", origin="lower")
        coronal_middiff.set_aspect(coronal_aspect_ratio)

        sagittal_middiff = plt.subplot(2, 3, 6)
        plt.imshow(img[midsagittal, :, :, middiff].T, cmap="gray", origin="lower")
        sagittal_middiff.set_aspect(sagittal_aspect_ratio)

    elif dim == 3 or (dim == 4 and img.shape[3] == 1):
        axial = plt.subplot(2, 3, 1)
        plt.imshow(img[:, :, midaxial, ...].T, cmap="gray", origin="lower")
        axial.set_aspect(axial_aspect_ratio)

        coronal = plt.subplot(2, 3, 2)
        plt.imshow(img[:, midcoronal, :, ...].T, cmap="gray", origin="lower")
        coronal.set_aspect(coronal_aspect_ratio)

        sagittal = plt.subplot(2, 3, 3)
        plt.imshow(img[midsagittal, :, :, ...].T, cmap="gray", origin="lower")
        sagittal.set_aspect(sagittal_aspect_ratio)

    elif dim == 2:
        plt.imshow(img.T, cmap="gray", origin="lower")

    plt.show()


def show_image_equalized(img, voxel_size=voxel_size):
    """
    This function shows the equalized mid-slices of MRI volumes
    with the correct coronal/sagittal aspect ratios.
    """
    import matplotlib.pyplot as plt
    from dipy.core.histeq import histeq

    dim = len(img.shape)
    if dim in (3, 4):
        midaxial = img.shape[2] // 2
        midcoronal = img.shape[1] // 2
        midsagittal = img.shape[0] // 2
        axial_aspect_ratio = voxel_size[1] / voxel_size[0]
        coronal_aspect_ratio = voxel_size[2] / voxel_size[0]
        sagittal_aspect_ratio = voxel_size[2] / voxel_size[1]

    if dim == 4 and img.shape[3] != 1:
        middiff = img.shape[3] // 2

        axial_b0 = plt.subplot(2, 3, 1)
        plt.imshow(histeq(img[:, :, midaxial, 0].astype('float')).T, cmap="gray", origin="lower")
        axial_b0.set_aspect(axial_aspect_ratio)

        coronal_b0 = plt.subplot(2, 3, 2)
        plt.imshow(histeq(img[:, midcoronal, :, 0].astype('float')).T, cmap="gray", origin="lower")
        coronal_b0.set_aspect(coronal_aspect_ratio)

        sagittal_b0 = plt.subplot(2, 3, 3)
        plt.imshow(histeq(img[midsagittal, :, :, 0].astype('float')).T, cmap="gray", origin="lower")
        sagittal_b0.set_aspect(sagittal_aspect_ratio)

        axial_middiff = plt.subplot(2, 3, 4)
        plt.imshow(histeq(img[:, :, midaxial, middiff].astype('float')).T, cmap="gray", origin="lower")
        axial_middiff.set_aspect(axial_aspect_ratio)

        coronal_middiff = plt.subplot(2, 3, 5)
        plt.imshow(histeq(img[:, midcoronal, :, middiff].astype('float')).T, cmap="gray", origin="lower")
        coronal_middiff.set_aspect(coronal_aspect_ratio)

        sagittal_middiff = plt.subplot(2, 3, 6)
        plt.imshow(histeq(img[midsagittal, :, :, middiff].astype('float')).T, cmap="gray", origin="lower")
        sagittal_middiff.set_aspect(sagittal_aspect_ratio)

    elif dim == 3 or (dim == 4 and img.shape[3] == 1):
        axial = plt.subplot(2, 3, 1)
        plt.imshow(histeq(img[:, :, midaxial, ...].astype('float')).T, cmap="gray", origin="lower")
        axial.set_aspect(axial_aspect_ratio)

        coronal = plt.subplot(2, 3, 2)
        plt.imshow(histeq(img[:, midcoronal, :, ...].astype('float')).T, cmap="gray", origin="lower")
        coronal.set_aspect(coronal_aspect_ratio)

        sagittal = plt.subplot(2, 3, 3)
        plt.imshow(histeq(img[midsagittal, :, :, ...].astype('float')).T, cmap="gray", origin="lower")
        sagittal.set_aspect(sagittal_aspect_ratio)

    elif dim == 2:
        plt.imshow(histeq(img.astype('float')).T, cmap="gray", origin="lower")

    plt.show()


def show_orientation(metaimage):
    from python_codes.nibabel import aff2axcodes
    print(aff2axcodes(metaimage.affine))








# DIPY basics
# -----------------------------------------------------------------------------
from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from python_codes.nibabel import flip_axis

import numpy as np

import os
from os.path import expanduser, join

# load the data:
home = expanduser('~')
path = join(home, 'Projects', 'neuroimaging', '../data/dipy')
os.chdir(path)
diff_path = join('subject1', 'dwi_orig.nii.gz')

img, affine, metaimage = load_nifti(diff_path, return_img=True)

print(img.shape)  # also can use: data.shape
voxel_size = metaimage.header.get_zooms()

# correct anterior-posterior orientation of the imported image:
# (these two methods do the same flips)

img2 = np.flip(img, 1)  # np.flip(img, axis=1)
# from nibabel.orientations import flip_axis
img3 = flip_axis(img, 1)  # flip_axis(img, axis=1)

show_image(img, voxel_size)
show_image(img2, voxel_size)
show_image(img3, voxel_size)  # img2 and img3 are the same
# show_orientation(metaimage)
# read b-values and b-vectors:

bvals, bvecs = read_bvals_bvecs('subject1/bvals', 'subject1/bvecs')
gtab = gradient_table(bvals, bvecs)
print(gtab.bvals)

# get the b0 image(s) using gtab b0 mask:
b0_img = img2[:, :, :, gtab.b0s_mask]
print(b0_img.shape)

# save the b0 image as a new nifti:
# affine is obviously wrong since we didn't change it when we flipped anterior/posterior
# save_nifti('subject1/PyCharm_outputs/b0.nii.gz', b0_img, affine)








# BASIC TRACTOGRAPHY
# -----------------------------------------------------------------------------
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data

import os

# download Stanford dataset:
# hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
# label_fname = get_fnames('stanford_labels')
# label_fname was: '/Users/Emad/.dipy/stanford_hardi/aparc-reduced.nii.gz'

# I moved the stanford_hardi folder to the project folder
# correct file names after moving data:
os.chdir('//dipy/stanford_hardi')
label_fname = 'aparc-reduced.nii.gz'
hardi_fname = 'HARDI150.nii.gz'
hardi_bval_fname = 'HARDI150.bval'
hardi_bvec_fname = 'HARDI150.bvec'

# load data:
img, affine, hardi_data = load_nifti(hardi_fname, return_img=True)
voxel_size = hardi_data.header.get_zooms()
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

# create a white-matter mask:
white_matter = (labels == 1) | (labels == 2)
show_image(white_matter)

# calculate fiber directions at all voxels of white matter using
# fitting the data to a constant solid angle ODF (orientation distribution function) at each voxel.
from dipy.reconst.csdeconv import auto_response
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model

response, ratio = auto_response(gtab, img, roi_radius=10, fa_thr=0.7)
csa_model = CsaOdfModel(gtab, sh_order=6)
csa_peaks = peaks_from_model(csa_model, img, default_sphere,
                             relative_peak_threshold=.8,
                             min_separation_angle=45,
                             mask=white_matter)

# visualize a slice from direction field:
from dipy.viz import window, actor, has_fury
if has_fury:
    ren = window.Scene()
    ren.add(actor.peak_slicer(csa_peaks.peak_dirs,
                              csa_peaks.peak_values,
                              colors=None))
    window.show(ren, size=(800, 800))
    # window.record(ren, out_path='csa_direction_field.png', size=(900, 900))

from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
stopping_criterion = ThresholdStoppingCriterion(csa_peaks.gfa, .25)

show_image(csa_peaks.gfa)
# thresholded generalized FA (gfa) > 0.25 image:
thresholded_gfa = csa_peaks.gfa > 0.25
show_image(thresholded_gfa)

# making the seed for tractography of corpus callosum:
'''
Here we use a 2×2×2 grid of seeds per voxel, 
in a sagittal slice of corpus callosum. 
Tracking from this region gives a model of the corpus callosum. 
This slice has label value 2 in the labels image.
'''
from dipy.tracking import utils
seed_mask = (labels == 2)
show_image(seed_mask)
seeds = utils.seeds_from_mask(seed_mask, affine, density=[2, 2, 2])

# Finally, tractography using EuDX algorithm:
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines

# Initialization of LocalTracking. The computation happens in the next step.
streamlines_generator = LocalTracking(csa_peaks, stopping_criterion, seeds,
                                      affine=affine, step_size=.5)
# Generate streamlines object
streamlines = Streamlines(streamlines_generator)

# Display the tracts:
from dipy.viz import colormap
if has_fury:
    # Prepare the display objects.
    color = colormap.line_colors(streamlines)
    streamlines_actor = actor.line(streamlines,
                                   colormap.line_colors(streamlines))
    # Create the 3D display.
    r = window.Renderer()
    r.add(streamlines_actor)
    window.show(r)
    # Save still images for this static example. Or for interactivity use
    # window.record(r, out_path='tractogram_EuDX.png', size=(800, 800))

# Save tractography as a TrackVis file:
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk

sft = StatefulTractogram(streamlines, hardi_data, Space.RASMM)
save_trk(sft, "tractogram_EuDX.trk", streamlines)









# GRADIENTS AND SPHERES
# -----------------------------------------------------------------------------
import numpy as np
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere

# make random points on a hemisphere using spherical polar coordinates:
n_pts = 64
theta = np.pi * np.random.rand(n_pts)
phi = 2 * np.pi * np.random.rand(n_pts)
hsph_initial = HemiSphere(theta=theta, phi=phi)     # hsph = hemisphere
# disperse_charges iteratively moves charges so that electrostatic potential energy is minimized
hsph_updated, potential = disperse_charges(hsph_initial, 5000)

# visualize points distributed on the hemisphere:

from dipy.viz import window, actor
ren = window.Scene()
ren.SetBackground(1, 1, 1)
ren.add(actor.point(hsph_initial.vertices, window.colors.red,
                    point_radius=0.05))
ren.add(actor.point(hsph_updated.vertices, window.colors.green,
                    point_radius=0.05))
window.show(ren)
# print('Saving illustration as initial_vs_updated.png')
# window.record(ren, out_path='initial_vs_updated.png', size=(300, 300))

# create a sphere from the hemisphere:
sph = Sphere(xyz=np.vstack((hsph_updated.vertices, -hsph_updated.vertices)))
# visualize the sphere:
window.rm_all(ren)
ren.add(actor.point(sph.vertices, window.colors.green, point_radius=0.05))
window.show(ren)
# print('Saving illustration as full_sphere.png')
# window.record(ren, out_path='full_sphere.png', size=(300, 300))

# create gradients from hsph_updated vectors:
from dipy.core.gradients import gradient_table
vertices = sph.vertices
values = np.ones(vertices.shape[0])
# We need 2 stacks of vertices, one for b-values of 1000 s/mm2 and one for 2500
bvecs = np.vstack((vertices, vertices))
bvals = np.hstack((1000 * values, 2500 * values))
# add two b0s at the beginning and end:
bvecs = np.insert(bvecs, (0, bvecs.shape[0]), np.array([0, 0, 0]), axis=0)
bvals = np.insert(bvals, (0, bvals.shape[0]), 0)
# create gradient table:
gtab = gradient_table(bvals, bvecs)
# visualize gradients:
window.rm_all(ren)
colors_b1000 = window.colors.blue * np.ones(vertices.shape)
colors_b2500 = window.colors.cyan * np.ones(vertices.shape)
colors = np.vstack((colors_b1000, colors_b2500))
colors = np.insert(colors, (0, colors.shape[0]), np.array([0, 0, 0]), axis=0)
colors = np.ascontiguousarray(colors)
ren.add(actor.point(gtab.gradients, colors, point_radius=100))
window.show(ren)
# print('Saving illustration as gradients.png')
# window.record(ren, out_path='gradients.png', size=(300, 300))










# BRAIN SEGMENTATION
# -----------------------------------------------------------------------------
import os
from os.path import join

import numpy as np

from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti

# scil_b0 dataset contains data for MR scanners:
# different companies and models. Here the data is for 1.5T Siemens.
# firs, download data:
data_fnames = get_fnames('scil_b0')
# then I moved the data to the project folder.
# load nifti image:
path = join('//dipy/datasets_multi-site_all_companies/1.5T/Siemens',
            'b0.nii.gz')
b0_img, affine = load_nifti(path)
b0_img = np.squeeze(b0_img)
show_image(b0_img)
# segment brain:
from dipy.segment.mask import median_otsu
b0_masked, brainmask = median_otsu(b0_img, median_radius=2, numpass=1)
# b0_masked_cropped, brainmask_cropped = median_otsu(data, median_radius=4, numpass=4,
#                                       autocrop=True)

# view brain segmentation results:
show_image_equalized(b0_img)
show_image_equalized(b0_masked)
show_image(brainmask)
# save segmentation results as nifti files (with dtype of float32):
os.chdir('//dipy/datasets_multi-site_all_companies/1.5T/Siemens')
save_nifti('b0_masked.nii.gz', b0_masked.astype(np.float32), affine)
save_nifti('brainmask.nii.gz', brainmask.astype(np.float32), affine)









# SNR estimation
# -----------------------------------------------------------------------------
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import TensorModel
import os

os.chdir('//dipy/stanford_hardi')
# load_nifti calls the metaimage the image! Hence return_img=True returns the metaimage.
img, affine, metaimage = load_nifti('HARDI150.nii.gz', return_img=True)
voxel_size = metaimage.header.get_zooms()
show_image(img)
bvals, bvecs = read_bvals_bvecs('HARDI150.bval', 'HARDI150.bvec')
gtab = gradient_table(bvals, bvecs)
b0_masked, brainmask = median_otsu(img, vol_idx=[0])
show_image_equalized(img)
show_image_equalized(b0_masked)
show_image(brainmask)

# compute tensors:
tenmodel = TensorModel(gtab)
tensorfit = tenmodel.fit(img, mask=brainmask)

# Compute worst-case/best-case SNR using corpus callosum:

from dipy.segment.mask import segment_from_cfa
from dipy.segment.mask import bounding_box
# set red-green-blue thresholds to (0.6, 1) in x axis and (0, 0.1) in y and z axes
# These values work well in practice to isolate the very RED voxels of the computed FA (cfa) map
thresholds = (0.6, 1, 0, 0.1, 0, 0.1)
# Let's define a box at the center of the brain that probably contains CC:
CCbox = np.zeros_like(img[..., 0])
mins, maxs = bounding_box(brainmask)
mins = np.array(mins)
maxs = np.array(maxs)
diff = (maxs - mins) // 4
bounds_min = mins + diff
bounds_max = maxs - diff
CCbox[bounds_min[0]:bounds_max[0],
       bounds_min[1]:bounds_max[1],
       bounds_min[2]:bounds_max[2]] = 1
# now we just want red voxels in CC:
CCmask, cfa = segment_from_cfa(tensorfit, CCbox, thresholds, return_cfa=True)
''' cfa is 4D; 4th dimension has 3 columns for red (commissural), green (association) and
blue (projection) fibers: '''
show_image(cfa[..., 0])     # commissural
show_image(cfa[..., 1])     # association
show_image(cfa[..., 2])     # projection
red = cfa[..., 0]
show_image(red)
show_image(CCmask)

# mean signal in CC
mean_signal = np.mean(img[CCmask], axis=0)

# background noise estimation
from scipy.ndimage.morphology import binary_dilation
backgroundmask = binary_dilation(brainmask, iterations=10)
show_image(brainmask)
show_image(backgroundmask)
# because we're interested in noise in air not signal in face / throat:
backgroundmask[..., :backgroundmask.shape[-1]//2] = 1
show_image(backgroundmask)
backgroundmask = ~backgroundmask
show_image(backgroundmask)
# save_nifti('backgroundmask.nii.gz', backgroundmask.astype(np.uint8), affine)
noise_std = np.std(img[backgroundmask, :])
print('Noise standard deviation= ', noise_std)

# now we can compute SNR for each diffusion orientation.
# for instance, let's compute SNR for diffusion orientations close to x, y, z directions:
b0indices = gtab.bvals == 0
# b0indices = np.sum(gtab.bvecs, axis=-1) == 0
gtab.bvecs[b0indices] = np.inf
axis_X = np.argmin(np.sum((gtab.bvecs - np.array([1, 0, 0]))**2, axis=-1))
axis_Y = np.argmin(np.sum((gtab.bvecs - np.array([0, 1, 0]))**2, axis=-1))
axis_Z = np.argmin(np.sum((gtab.bvecs - np.array([0, 0, 1]))**2, axis=-1))

for direction in [0, axis_X, axis_Y, axis_Z]:
    SNR = mean_signal[direction]/noise_std
    if direction == 0:
        print("SNR for the b=0 image is :", SNR)
    else:
        print("SNR for direction", direction, " ",
              gtab.bvecs[direction], "is :", SNR)










# Denoising with non-local means (NLMEANS)
# -----------------------------------------------------------------------------
import numpy as np
from time import time
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.io.image import load_nifti
import os

# dwi_fname, dwi_bval_fname, dwi_bvec_fname = get_fnames('sherbrooke_3shell')
# after copying data into dipy folder:
os.chdir('//dipy/sherbrooke_3shell')
img, affine, metaimage = load_nifti('HARDI193.nii.gz', return_img=True)
voxel_size = metaimage.header.get_zooms()
mask = img[..., 0] > 80
# We select only one volume for the example to run quickly.
img2 = img[..., 1]
'''
To call non_local_means we have to to estimate the standard deviation of the noise first. 
We use N=4 since the Sherbrooke scan was acquired on 1.5T Siemens with a 4 array head coil.
'''
sigma = estimate_sigma(img2, N=4)

t = time()
img2_denoised = nlmeans(img2, sigma=sigma, mask=mask, patch_radius=1,
              block_radius=1, rician=True)
print("Processing time", time() - t)

img2_difference = np.abs(img2_denoised.astype(np.float64) - img2.astype(np.float64))
img2_difference[~mask] = 0
show_image(img2)
show_image(img2_denoised)
show_image(img2_difference)
# nib.save(nib.Nifti1Image(img2_denoised, affine), 'vol1_denoised.nii.gz')








# Denoising with Local PCA via empirical thresholds
# -----------------------------------------------------------------------------
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.denoise.localpca import localpca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
import os

# dwi_fname, dwi_bval_fname, dwi_bvec_fname = get_fnames('isbi2013_2shell')
os.chdir('//dipy/isbi2013')
img, affine, metaimage = load_nifti('phantom64.nii.gz', return_img=True)
voxel_size = metaimage.header.get_zooms()
bvals, bvecs = read_bvals_bvecs('phantom64.bval', 'phantom64.bvec')
gtab = gradient_table(bvals, bvecs)
print("Image shape: ", img.shape)
'''
pca_noise_estimate gives sigma, to be used in local PCA algorithm. 
It takes img and gradient table as inputs, and returns 
estimate of local noise standard deviation as a 3D array. 
We return a smoothed version, where a Gaussian filter with radius 3 voxels has been applied 
to the estimate of the noise before returning it. 
We correct for the bias due to Rician noise.
'''
sigma = pca_noise_estimate(img, gtab, correct_bias=True, smooth=3)
'''
localpca takes into account the multi-dimensional diffusion MRI data. 
It performs PCA on local 4D patch and then removes the noise components 
by thresholding the lowest eigenvalues. The eigenvalue threshold will be computed 
from the local variance estimate performed by the pca_noise_estimate, 
if this is inputted in the main localpca function. The relationship between 
the noise variance estimate and the eigenvalue threshold can be adjusted using 
the input parameter tau_factor. According to Manjon et al. [Manjon2013], 
this parameter is set to 2.3.
'''
# beware: this takes time!
img_denoised = localpca(img, sigma, tau_factor=2.3, patch_radius=2)
img_rms_difference = np.sqrt((img_denoised - img)**2)
# viewing results:
show_image(img)
show_image(img_denoised)
show_image(img_rms_difference)
# save denoised image as a nifti file:
# nib.save(nib.Nifti1Image(img_denoised, affine), 'img_denoised.nii.gz')









# Denoise using Marcenko-Pastur PCA algorithm
# -----------------------------------------------------------------------------
'''
PCA-based denoising exploits redundancy across diffusion images [Manjon2013], [Veraart2016a]. 
This algorithm provides optimal compromise between noise suppression and 
loss of anatomical information for different techniques such as DTI [Manjon2013], 
spherical deconvolution [Veraart2016a] and DKI [Henri2018].

Basic idea behind PCA-based denoising is to remove components of data that are classified 
as noise. Principal Components classification can be performed based on 
prior noise variance estimates [Manjon2013] (see denoise_localpca) or 
automatically based on the Marcenko-Pastur distribution [Veraa2016a]. 
In addition to noise suppression, the PCA algorithm can be used to get the standard deviation 
of the noise [Veraa2016b].

In the following example, we show how to denoise diffusion MRI and 
estimate the noise standard deviation using the PCA based on the Marcenko-Pastur distribution.
'''
# load general modules
import numpy as np
import matplotlib.pyplot as plt
from time import time

# load pca function using Marcenko-Pastur distribution
from dipy.denoise.localpca import mppca
# load other dipy's functions that will be used for analysis
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.segment.mask import median_otsu
import dipy.reconst.dki as dki

# load functions to fetch data for this example
import os

# load data:
# dwi_fname, dwi_bval_fname, dwi_bvec_fname, _ = get_fnames('cfin_multib')
# after copying data into projects's dipy folder and renaming files:
os.chdir('//dipy/cfin_multib')
img, affine, metaimage = load_nifti('diffImage.nii', return_img=True)
voxel_size = metaimage.header.get_zooms()
bvals, bvecs = read_bvals_bvecs('diffImage.bval', 'diffImage.bvec')
gtab = gradient_table(bvals, bvecs)

# For simplicity, we only select 2 non-zero b-values for this example:
vol_indices = (bvals == 0) | (bvals == 1000) | (bvals == 2000)
img2 = img[..., vol_indices]
gtab2 = gradient_table(bvals[vol_indices], bvecs[vol_indices])

# PCA denoising using the Marcenko-Pastur distribution:
# sigma: a volume showing noise std across all volumes
t = time()
img2_denoised, sigma = mppca(img2, patch_radius=2, return_sigma=True)
print("Time for processing local MP-PCA: ", time() - t)
'''
The mppca algorithm denoises diffusion data using a 3D sliding window 
which is defined by the variable patch_radius. This window should comprise a larger number 
of voxels than the number of diffusion volumes. Since our data has 67 volumes, 
the patch_radius is set to 2 which corresponds to a 5x5x5 sliding window comprising 125 voxels.

To assess the performance of the Marcenko-Pastur PCA denosing, an axial slice of the original 
data, denoised data, and residuals are plotted below.
'''
img2_rms_difference = np.sqrt((img2_denoised - img2)**2)
show_image(img2)
show_image(img2_denoised)
show_image(img2_rms_difference)
# nib.save(nib.Nifti1Image(img2_denoised, affine), 'img2_denoised.nii.gz')
'''
Difference image shows only random noise, indicating that the data’s structural information 
is preserved by the PCA denoising algorithm.
Additionally, we show how the PCA denoising algorithm affects different diffusion measurements. 
For this, we run the diffusion kurtosis model below on both original and denoised :
'''
dkimodel = dki.DiffusionKurtosisModel(gtab2)
img2_masked, mask = median_otsu(img2, vol_idx=[0, 1], median_radius=4, numpass=2,
                             autocrop=False, dilate=1)
dki_orig = dkimodel.fit(img2, mask=mask)
dki_denoised = dkimodel.fit(img2_denoised, mask=mask)
# FA, mean diffusivity, and mean kurtosis:
FA_orig = dki_orig.fa
FA_denoised = dki_denoised.fa
MD_orig = dki_orig.md
MD_denoised = dki_denoised.md
MK_orig = dki_orig.mk(0, 3)
MK_denoised = dki_denoised.mk(0, 3)

# visualize results:
midaxial = img2.shape[2] // 2
fig2, ax = plt.subplots(2, 3, figsize=(10, 6), subplot_kw={'xticks': [], 'yticks': []})
fig2.subplots_adjust(hspace=0.3, wspace=0.03)
ax.flat[0].imshow(MD_orig[:, :, midaxial].T, cmap='gray', vmin=0, vmax=2.0e-3, origin='lower')
ax.flat[0].set_title('MD (DKI)')
ax.flat[1].imshow(FA_orig[:, :, midaxial].T, cmap='gray', vmin=0, vmax=0.7, origin='lower')
ax.flat[1].set_title('FA (DKI)')
ax.flat[2].imshow(MK_orig[:, :, midaxial].T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[2].set_title('AD (DKI)')
ax.flat[3].imshow(MD_denoised[:, :, midaxial].T, cmap='gray', vmin=0, vmax=2.0e-3, origin='lower')
ax.flat[3].set_title('MD (DKI)')
ax.flat[4].imshow(FA_denoised[:, :, midaxial].T, cmap='gray', vmin=0, vmax=0.7, origin='lower')
ax.flat[4].set_title('FA (DKI)')
ax.flat[5].imshow(MK_denoised[:, :, midaxial].T, cmap='gray', vmin=0, vmax=1.5, origin='lower')
ax.flat[5].set_title('AD (DKI)')
plt.show()
# fig2.savefig('denoised_dki.png')

# visualizing noise standard deviation (calculated using Marcenko-Pastur PCA algorithm above):
show_image(sigma)
# mean noise across all voxels:
sigma_mean = np.mean(sigma[mask])
print(sigma_mean)
# we  can use sigma_mean to compute image’s nominal SNR (i.e. SNR at b0):
b0_denoised = img2_denoised[..., 0]
b0_mean_signal = np.mean(b0_denoised[mask])
snr_b0_denoised = b0_mean_signal / sigma_mean
print(snr_b0_denoised)








# Supperssing Gibbs oscillations
# -----------------------------------------------------------------------------
from dipy.denoise.gibbs import gibbs_removal
from dipy.io.image import load_nifti_data
import numpy as np
from matplotlib import pyplot as plt
import os

# t1_fname, t1_denoised_fname, ap_fname = get_fnames('tissue_data')
os.chdir('//dipy/tissue_data')
t1 = load_nifti_data('t1_brain.nii.gz')
show_image(t1)
'''
Due to the high quality of this data, Gibbs artefacts are not visually evident in this image. 
Therefore, to analyse the benefits of the Gibbs suppression algorithm, 
Gibbs artefacts are artificially introduced by removing high frequencies of the image’s 
Fourier transform:
'''
midaxial = t1.shape[-1] // 2
t1_slice = t1[..., midaxial]
t1_slice_fourier = np.fft.fft2(t1_slice)
t1_slice_fourier_zerocenter = np.fft.fftshift(t1_slice_fourier)
t1_slice_fourier_cropped = t1_slice_fourier_zerocenter[64: 192, 64: 192]
t1_slice_gibbs = abs(np.fft.ifft2(t1_slice_fourier_cropped)/4)
show_image(t1_slice)
show_image(t1_slice_gibbs)

# Gibbs artefact removal:
t1_slice_unring = gibbs_removal(t1_slice_gibbs)
t1_slice_difference = t1_slice_unring - t1_slice_gibbs

# Plot results:
'''
You can see that artefactual oscillations are visually suppressed without compromising 
the contrast between white and grey matter (green arrow on the middle image). 
'''
fig1, ax = plt.subplots(1, 3, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})
ax.flat[0].imshow(t1_slice_gibbs.T, cmap="gray", vmin=100, vmax=400, origin="lower")
ax.flat[0].annotate('Rings', fontsize=10, xy=(81, 70),
                    color='red',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))
ax.flat[1].imshow(t1_slice_unring.T, cmap="gray", vmin=100, vmax=400, origin="lower")
ax.flat[1].annotate('WM/GM', fontsize=10, xy=(78, 75),
                    color='green',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='green'))
ax.flat[2].imshow(t1_slice_difference.T, cmap="gray", vmin=0, vmax=10, origin="lower")
ax.flat[2].annotate('Rings', fontsize=10, xy=(81, 70),
                    color='red',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))
plt.show()
# fig1.savefig('Gibbs_suppression_structural.png')

# Not let's do Gibbs removal on diffusion images.
# first, load noisy data provided by CENIR, ICM, Paris:
from dipy.data import read_cenir_multib
bvals = [200, 400, 1000, 2000]
metaimage, gtab = read_cenir_multib(bvals)
img = np.asarray(metaimage.dataobj)
show_image(img)
# for this example, we take 2 slices of the image:
img_slice = img[:, :, 40:42, :]
show_image(img_slice)

img_slice_corrected = gibbs_removal(img_slice, slice_axis=2)
img_slice_difference = img_slice_corrected - img_slice
show_image(img_slice_corrected)
show_image_equalized(img_slice_difference)

'''
The above figures shows that suppressed Gibbs artefacts are hard to discern on b0 image. 
Therefore, diffusion metrics for both uncorrected and corrected img are computed using 
mean signal diffusion kurtosis imaging (MSDKI):
'''
from dipy.segment.mask import median_otsu
img_masked, brainmask = median_otsu(img_slice, vol_idx=range(10, 50),
                             median_radius=3, numpass=1, autocrop=False,
                             dilate=1)
# Define mean signal diffusion kurtosis model
import dipy.reconst.msdki as msdki
dki_model = msdki.MeanDiffusionKurtosisModel(gtab)
# Fit the uncorrected data
dki_fit = dki_model.fit(img_slice, mask=brainmask)
MSK_orig = dki_fit.msk
# Fit the corrected data
dki_fit = dki_model.fit(img_slice_corrected, mask=brainmask)
MSK_corrected = dki_fit.msk

# visualize results:
fig3, ax = plt.subplots(1, 3, figsize=(12, 12),
                        subplot_kw={'xticks': [], 'yticks': []})
ax.flat[0].imshow(MSK_orig[:, :, 0].T, cmap='gray', origin='lower',
                  vmin=0, vmax=1.5)
ax.flat[0].set_title('MSK (uncorrected)')
ax.flat[0].annotate('Rings', fontsize=12, xy=(59, 63),
                    color='red',
                    xycoords='data', xytext=(45, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))
ax.flat[1].imshow(MSK_corrected[:, :, 0].T, cmap='gray', origin='lower',
                  vmin=0, vmax=1.5)
ax.flat[1].set_title('MSK (corrected)')
ax.flat[2].imshow(MSK_corrected[:, :, 0].T - MSK_orig[:, :, 0].T, cmap='gray',
                  origin='lower', vmin=-0.2, vmax=0.2)
ax.flat[2].set_title('MSK (uncorrected - corrected')
ax.flat[2].annotate('Rings', fontsize=12, xy=(59, 63),
                    color='red',
                    xycoords='data', xytext=(45, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))
plt.show()
# fig3.savefig('Gibbs_suppression_msdki.png')









# Reslice diffusion images (to make isotropic voxel)
# -----------------------------------------------------------------------------
from dipy.align.reslice import reslice
from dipy.data import get_fnames
from dipy.io.image import load_nifti
import numpy as np

img_path = get_fnames('aniso_vox')
img, affine, voxel_size = load_nifti(img_path, return_voxsize=True)
print(img.shape)
print(voxel_size)
show_image(img)
img2 = np.flip(img, axis=1)     # correcting anterior-posterior orientation
M = np.reshape([1, 0, 0, 0,     # alternatively: M = np.eye(4); M[1,1] = -1
                0, -1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1], (4, 4))

affine2 = M @ affine
show_image(img2, voxel_size)

new_voxel_size = (3, 3, 3)
img3, affine3 = reslice(img2, affine2, voxel_size, new_voxel_size)  # interpolation is trilinear by default
print(img3.shape)
show_image(img3, new_voxel_size)
# save_nifti('iso_vox.nii.gz', img3, affine3)
# or save as analyze format for SPM analysis:
# metaimage3 = nib.Spm2AnalyzeImage(img3, affinee)
# nib.save(metaimage3, 'iso_vox.img')









# Reconstruction with Constrained Spherical Deconvolution
# -----------------------------------------------------------------------------
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
# hardi image has 10 b0 volumes and 150 volumes of b-2000.
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
img, affine, voxel_size = load_nifti(hardi_fname, return_voxsize=True)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

# todo: complete fCSD :)











