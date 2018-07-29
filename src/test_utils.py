# Imports
from __future__ import division
import numpy as np
import os, sys, shutil, re
import random, time
import scipy.ndimage as snd
import random
from datetime import datetime
import glob
import matplotlib.pyplot as plt
import SimpleITK as sitk

sys.path.append("../")
from helpers.utils import imshow
rng = np.random.RandomState(40)


import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd 
from collections import OrderedDict
import nibabel as nib

HEADER = ["Name", "dice", "jaccard", "Hausdorff"]

def save_nii(vol, affine, hdr, path, prefix, suffix):
    vol = nib.Nifti1Image(vol, affine, hdr)
    vol.set_data_dtype(np.uint8)
    nib.save(vol, os.path.join(path, prefix+'_'+suffix))

def load_nii(img_file, folder_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header
    """
    nimg = nib.load(os.path.join(folder_path, img_file))
    return nimg.get_data(), nimg.affine, nimg.header
 
def sitk_show(nda, title=None, margin=0.0, dpi=40):
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
 
    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
 
    plt.set_cmap("gray")
    for k in range(0,nda.shape[2]):
        print "printing slice "+str(k)
        ax.imshow(np.squeeze(nda[:,:,k]),extent=extent,interpolation=None)
        plt.draw()
        plt.pause(0.1)
        #plt.waitforbuttonpress()

def get_test_data_path(path, gt_available=False):
	Flair = []
	T1 = []
	T2 = []
	T1c = []
	Mask = []
	folders = []
	Truth = []
	Subdir_array = []
	IMAGE_LIST = []
	GROUND_LIST = []

	for subdir, dirs, files in os.walk(path):
	    for file1 in files:
	        if file1[-6:-3]=='nii' and ('flair' in file1):
	            Flair.append(file1)
	            folders.append(subdir+'/')
	            Subdir_array.append(subdir[-5:])
	        elif file1[-6:-3]=='nii' and ('t1' in file1 and 't1ce' not in file1):
	            T1.append(file1)
	        elif file1[-6:-3]=='nii' and ('t2' in file1 and 'double_spie_f_t2_one_nd.nii' not in file1):
	            T2.append(file1)
	        elif file1[-6:-3]=='nii' and 'mask' in file1:
	            Mask.append(file1)
	        elif file1[-6:-3]=='nii' and ('t1ce' in file1 or 't1_ce' in file1 ):
	                T1c.append(file1)
	        elif file1[-6:-3]=='nii' and 'seg' in file1:
	            Truth.append(file1)

	for i in xrange(len(Flair)):
		# if i > 2:
		# 	break

		flair_volume = folders[i]+ Flair[i]
		t2_volume    = folders[i]+ T2[i]
		t1_volume    = folders[i]+ T1[i]
		t1ce_volume  = folders[i]+T1c[i]

		img_list     = [flair_volume, t1_volume, t2_volume,t1ce_volume]
		IMAGE_LIST.append(img_list)

		if gt_available:
			truth_volume = folders[i]+Truth[i]
			ground_list  = [truth_volume]
			GROUND_LIST.append(ground_list)

	return IMAGE_LIST, GROUND_LIST  
	 
def get_patient_data(sequence_path, gt_path):
	patient_dict= {}
	corres_flair = nib.load(sequence_path[0])
	corres_t1    = nib.load(sequence_path[1])
	corres_t2    = nib.load(sequence_path[2])
	corres_t1c   = nib.load(sequence_path[3])


	corres_affine= corres_flair.get_affine()

	corres_flair = corres_flair.get_data()
	corres_t1    = corres_t1.get_data()
	corres_t2    = corres_t2.get_data()
	corres_t1c   = corres_t1c.get_data()

	if gt_path:
		corres_truth = nib.load(gt_path[0])
		corres_truth = corres_truth.get_data()
		patient_dict['Truth'] = corres_truth

	number_of_slices = corres_flair.shape[2]
	patient_dict['FLAIR']            = corres_flair 
	patient_dict['T1']               = corres_t1
	patient_dict['T2']               = corres_t2
	patient_dict['T1c'] 			 = corres_t1c
	patient_dict['Affine']           = corres_affine
	return patient_dict

