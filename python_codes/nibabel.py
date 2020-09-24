import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Set numpy to print 3 decimal points and suppress small values
np.set_printoptions(precision=2, suppress=True)


epi = nib.load('someones_epi.nii.gz')
epivol = epi.get_fdata()
epi_dimensions = epi.shape
epi_datatype = epi.get_data_dtype()
epi_voxelsize = epi.header.get_zooms()


def show_slices(slices):
    """ Function to display image slices passed to it """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


slice_0 = epivol[26, :, :]
slice_1 = epivol[:, 30, :]
slice_2 = epivol[:, :, 16]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")
plt.show()


anat = nib.load('someones_anatomy.nii.gz')
anatvol = anat.get_fdata()
print(anatvol.shape)
show_slices([anatvol[28, :, :],
             anatvol[:, 33, :],
             anatvol[:, :, 28]])
plt.suptitle("Center slices for anatomical image")
plt.show()

print(epi.affine)

# Let's decompose the epi image affine transform (from scanner space to epi space)
# into M matrix (rotation, zoom and shear) and abc vector (translation):
M = epi.affine[:3, :3]
abc = epi.affine[:3, 3]


def afftrans(M, abc, i, j, k):
    return M.dot([i, j, k]) + abc


# Now let's see what's the coordinantes of epi image epicenter in scanner space
epi_center = (np.array(epivol.shape) - 1) / 2

epi_center_coordinates1 = afftrans(M, abc,
                                   epi_center[0],
                                   epi_center[1],
                                   epi_center[2])

epi_center_coordinates2 = epi.affine.dot(list(epi_center) + [1])

# nibabel's apply_affine function:
from python_codes.nibabel import apply_affine

epi_center_coordinates3 = apply_affine(epi.affine, epi_center)

# Matrix inversion on the anatomical affine to map between epi and anatomical image:
import numpy.linalg as npl

T_epi2anat = npl.inv(anat.affine).dot(epi.affine)
# coordinates of epi epicenter in the anatomic space:
epi_center_in_anat = apply_affine(T_epi2anat, epi_center)
# coordinates of anatomic epicenter in scanner space:
anat_center = (np.array(anatvol.shape) - 1) / 2

# Let's practice with affine transforms a bit!
Tscale = np.array([[3, 0, 0, 0],
                   [0, 3, 0, 0],
                   [0, 0, 3, 0],
                   [0, 0, 0, 1]])
alpha = 0.3
cosa = np.cos(alpha)
sina = np.sin(alpha)
Trotate = np.array([[1, 0, 0, 0],
                    [0, cosa, -sina, 0],
                    [0, sina, cosa, 0],
                    [0, 0, 0, 1]])
Tsclrot = Tscale.dot(Trotate)
Ttranslate = np.array([[1, 0, 0, -78],
                       [0, 1, 0, -76],
                       [0, 0, 1, -64],
                       [0, 0, 0, 1]])
Twhole = Ttranslate.dot(Tsclrot)


# epi image proxy (i.e. points to epi image without loading it from disk yet):
nib.is_proxy(epi.dataobj)

# Creating a nifti image from an array using nibable:

mat = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
affine = np.diag([1, 2, 3, 1])
newnifti = nib.Nifti1Image(mat, affine)
'''
Here the newnifti data is already a numpy array, 
and there is no version of the array on disk. 
The dataobj property of newnifti is the array itself 
rather than a proxy for the array:
'''
print(newnifti.dataobj)
print(newnifti.dataobj is mat)
print(nib.is_proxy(newnifti.dataobj))
# Compare with:
print(epi.dataobj)
print(epi.dataobj is epivol)
print(nib.is_proxy(epi.dataobj))


# Image slicing

epi2 = epi.slicer[2:50, ...]
print(epi.shape, epi2.shape)
# this would be True:
np.array_equal(epi2.get_fdata(), epi.get_fdata()[2:50, ...])
# affine attribute of epi2 is also corrected automatically:
print(epi.affine, '\n', epi2.affine)
affine_diff = epi.affine - epi2.affine

# Downsampling using slicer attribute (causes artefacts in frequency domain):
from matplotlib.pyplot import imshow

anat_downsampled = anat.slicer[::2, ::2, ::2]
print(anat.header.get_zooms(), anat_downsampled.header.get_zooms())

imshow(anat.get_fdata()[..., 30].T, cmap='gray', origin='lower')
plt.show()
imshow(anat_downsampled.get_fdata()[..., 15].T, cmap='gray', origin='lower')
plt.show()

# Flipping the image along an axis (affine attributes updates automatically)

anat_flipped = anat.slicer[::-1]
print(nib.orientations.aff2axcodes(anat.affine))
print(nib.orientations.aff2axcodes(anat_flipped.affine))
print(anat.affine, '\n', anat_flipped.affine)

# Saving nifti volumes on disk

nib.save(anat_flipped, 'someones_anatomy_flipped.nii')
# or
anat_flipped.to_filename('someones_anatomy_flipped2.nii')
# setting and getting filename:
anat_flipped.set_filename('someones_anatomy_flipped.nii')
anat_flipped.get_filename()
anat_flipped.to_filename(anat_flipped.get_filename())
# nifti.file_map is a dictionary:
# keys are the names of the files used to load / save on disk
print(anat.file_map['image'].filename)




# Turn dicoms into 3D volume:

path = '...'
imageList = os.listdir(path)
slices = [dicom.read_file(path + imageName, force=True)
          for imageName in imageList]

# this step is really important: it sorts the slices.
slices = sorted(slices, key=lambda x:x.ImagePositionPatient[2])


pixel_spacing = slices[0].PixelSpacing
slice_thickness = slices[0].SliceThickness
axial_aspect_ratio = pixel_spacing[1] / pixel_spacing[0]
sagittal_aspect_ratio = pixel_spacing[1] / slice_thickness
coronal_aspect_ratio = slice_thickness / pixel_spacing[0]
image_shape = list(slices[0].pixel_array.shape)
image_shape.append(len(slices))

volume3d = np.zeros(image_shape)

for i, s in enumerate(slices):
    array2d = s.pixel_array
    volume3d[:, :, i] = array2d