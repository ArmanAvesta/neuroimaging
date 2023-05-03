trac-all -prep -c $SUBJECTS_DIR/_Scripts/dmrirc11.txt -no-isrunning
trac-all -bedp -c $SUBJECTS_DIR/_Scripts/dmrirc11.txt -no-isrunning
trac-all -path -c $SUBJECTS_DIR/_Scripts/dmrirc11.txt -no-isrunning


trac-all -stat -c $SUBJECTS_DIR/_Scripts/dmrircStat.txt



#corrected dmrilowb_brain.nii.gz

trac-all -nocorr -noqa -c dmrirc_lowbcorrect.txt
trac-all -bedp -c dmrirc_lowbcorrect.txt
trac-all -path -c dmrirc_lowbcorrect.txt