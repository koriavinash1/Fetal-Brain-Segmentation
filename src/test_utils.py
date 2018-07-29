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


# https://programtalk.com/vs2/?source=python/4202/VNet/VNet.py
def computeQualityMeasures(lP, lT, class_label):
    # Get a SimpleITK Image from a numpy array. 
    # If isVector is True, then a 3D array will be treaded 
    # as a 2D vector image, otherwise it will be treaded as a 3D image

	quality=OrderedDict()
	labelPred=sitk.GetImageFromArray(lP, isVector=False)
	labelTrue=sitk.GetImageFromArray(lT, isVector=False)

	dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
	dicecomputer.Execute(labelTrue==class_label,labelPred==class_label)
	quality["dice"]=dicecomputer.GetDiceCoefficient()
	quality["jaccard"]=dicecomputer.GetJaccardCoefficient()
	#    quality["dice"]=dicecomputer.GetMeanOverlap()
	# quality["dice"]=dicecomputer.GetVolumeSimilarity() 
	# quality["dice"]=dicecomputer.GetUnionOverlap()
	# quality["dice"]=dicecomputer.GetFalseNegativeError() 
	#    quality["dice"]=dicecomputer.GetFalsePositiveError () 
	# Check if both the images have non-zero pixel count?
	# Else it will throw error. Just set 0 distance if pixel count =0
	quality["avgHausdorff"]=0
	quality["Hausdorff"]=0
	# Disable Hausdorff: Takes long time to compute 
	# # READ:
	# # https://itk.org/Doxygen/html/classitk_1_1DirectedHausdorffDistanceImageFilter.html
	# # https://itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1HausdorffDistanceImageFilter.html#a0bc838ff0d5624132abdbe089eb54705
	try:		
		if (np.count_nonzero(labelTrue) and np.count_nonzero(labelPred)):
			hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
			hausdorffcomputer.Execute(labelTrue==class_label,labelPred==class_label)
			quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance()
			quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()
	except Exception,e:
		print str(e)
	return quality
 

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
 
    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """
 
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
 
    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
 
    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
 
    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    #interp_t_values = np.zeros_like(source,dtype=float)
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
 
    return interp_t_values[bin_idx].reshape(oldshape)
 
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


if __name__ == '__main__':
	pred_path = '/home/mahendrakhened/Python_Projects/PhD_IIT_M/2017/LV_2011/models/ResidualDenseNet/predictions20171022_232049/DET0045301'
	seg_path  = '/home/mahendrakhened/Python_Projects/PhD_IIT_M/2017/LV_2011/processed_dataset/dataset/test_set/DET0045301'
	pred_files_path_list =  glob.glob(pred_path + "/*_SA*_ph*.png")
	seg_files_path_list = glob.glob(seg_path + "/*_SA*_ph*.png")
	calcSegmentationMetrics(pred_files_path_list, seg_files_path_list, extension='.png', result_path = './')