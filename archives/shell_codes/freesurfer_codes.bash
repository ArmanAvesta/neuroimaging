
#——————————————————————————————
#FreeSurfer Setup
export FREESURFER_HOME=/Applications/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export NO_FSFAST=1

FREESURFER_HOME=/Applications/freesurfer
FSFAST_HOME=/Applications/freesurfer/fsfast
FSF_OUTPUT_FORMAT=nii
MNI_DIR=/Applications/freesurfer/mni

#SUBJECTS_DIR=/Users/Emad/Projects/FreeSurfer/subjects
SUBJECTS_DIR=/Users/Emad/Projects/FreeSurfer/subjects



# FSL Setup
FSLDIR=/Applications/FSL/fsl
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH
. ${FSLDIR}/etc/fslconf/fsl.sh




#——————————————————————————————————————————
#Project Functions

adni() {
export SUBJECTS_DIR=/Volumes/EmadArtIntel/ADNI/SubjectsDir
cd $SUBJECTS_DIR
}


#——————————————————————————————————————————
#Function Initialization

adni



#——————————————————————————————
#FreeSurfer Custom Functions

fshow() {
echo volshow
echo imshow
echo difshow
echo pathshow
echo surfshow
echo surflineshow
echo thickshow
echo parcshow
echo surfshowall
}

volshow() {
cd $SUBJECTS_DIR/$1
freeview -v mri/T1.mgz \
mri/aparc+aseg.mgz:colormap=lut
cd $SUBJECTS_DIR
}

imshow() {
cd $SUBJECTS_DIR/$1
freeview -v mri/T1.mgz \
mri/aparc+aseg.mgz:colormap=lut \
& \
freeview -v dlabel/diff/anat_brain_mask.flt.nii.gz \
dmri/dtifit_FA.nii.gz:colormap=heat
cd $SUBJECTS_DIR
}

difshow() {
cd $SUBJECTS_DIR/$1
freeview -v dmri/dwi.nii.gz
cd $SUBJECTS_DIR
}

pathshow() {
cd $SUBJECTS_DIR/$1
freeview -v dlabel/diff/anat_brain_mask.flt.nii.gz:colormap=heat \
-tv dpath/merged_avg33_mni_flt.mgz
cd $SUBJECTS_DIR
}

surfshow() {
cd $SUBJECTS_DIR/$1
freeview -f \
surf/lh.inflated:visible=0 \
surf/lh.white:visible=0 \
surf/lh.pial \
surf/rh.inflated:visible=0 \
surf/rh.white:visible=0 \
surf/rh.pial
cd $SUBJECTS_DIR
}

surflineshow() {
cd $SUBJECTS_DIR/$1
freeview -v mri/brain.mgz \
-f surf/lh.white:edgecolor=red surf/rh.white:edgecolor=red \
surf/lh.pial surf/rh.pial
cd $SUBJECTS_DIR
}

thickshow() {
cd $SUBJECTS_DIR/$1
freeview -f \
surf/lh.inflated:overlay=lh.thickness:overlay_threshold=0.1,3:name=lh.thickness \
surf/rh.inflated:overlay=rh.thickness:overlay_threshold=0.1,3:name=rh.thickness
cd $SUBJECTS_DIR
}

parcshow() {
cd $SUBJECTS_DIR/$1
freeview -f  surf/lh.pial:annot=aparc.annot:name=lh.parc \
surf/rh.pial:annot=aparc.annot:name=rh.parc
cd $SUBJECTS_DIR
}


surfshowall() {
cd $SUBJECTS_DIR/$1
freeview -v mri/brain.mgz \
-f surf/lh.white:edgecolor=red surf/rh.white:edgecolor=red \
surf/lh.pial surf/rh.pial \
& \
freeview -f  surf/lh.pial:annot=aparc.annot:name=lh.parc:visible=0 \
surf/lh.inflated:overlay=lh.thickness:overlay_threshold=0.1,3:name=lh.thickness:visible=0 \
surf/lh.inflated:visible=0 \
surf/lh.white:visible=0 \
surf/lh.pial \
surf/rh.pial:annot=aparc.annot:name=rh.parc:visible=0 \
surf/rh.inflated:overlay=rh.thickness:overlay_threshold=0.1,3:name=rh.thickness:visible=0 \
surf/rh.inflated:visible=0 \
surf/rh.white:visible=0 \
surf/rh.pial
cd $SUBJECTS_DIR
}


