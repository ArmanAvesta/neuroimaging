for subj in case_list ; do
	recon-all -i $RAW_DATA/$subj/T1/IM-0001-0001.dcm -s $subj -all
done