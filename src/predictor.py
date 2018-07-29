from __future__ import division
import numpy as np
import os, shutil, sys
import SimpleITK as sitk
import glob
from datetime import datetime
import time
from estimator import *
from config import *
from networks import *
from test_utils import *
from train_utils import *


      	
if __name__ == "__main__":
	# Set Environment
	conf = conf()
	model_path = os.path.join(conf.output_dir, conf.run_name, 'best_model/latest.ckpt')
	test_data = 'path/to/test/data'

	patient_folders = next(os.walk(test_data))[1]
	save_dir = os.path.join(conf.output_dir, conf.run_name, 'predictions{}'.format(time.strftime("%Y%m%d_%H%M%S")))
	# if os.path.exists(save_dir):
	#     shutil.rmtree(save_dir)
	# os.makedirs(save_dir)
	# for patient in patient_folders:
	# 	os.makedirs(os.path.join(save_dir, patient))

	inputs = getInputsPlaceholder(None,None,None,conf.num_channels)
	targets = getTargetsPlaceholder(None,None,None,conf.num_class)
	weight_maps = tf.placeholder(tf.float32, shape=[None,None,None])
	batch_class_weights = tf.placeholder(tf.float32)

	# define the net			
	print('Defining the network')
	net = 	UNET(inputs,
			targets, 
			weight_maps,
			num_class=conf.num_class,
			n_pool=3,
			n_feat_first_layer=[32],
			chief_class = conf.chief_class,
			weight_decay = 5e-6,
			metrics_list = ['plain_dice_loss', 'dice_score_class_1', 'dice_score_class_0', 'dice_score'],
			metric_to_optimize = 'plain_dice_loss',
			optimizer = Adam(1e-4),
			gpu_ids = [1])

	# initialise the estimator with the net
	print('Preparing the estimator..')
	trainer = Estimator(net_obj = net,
						summary_dir = '',
						freq_list = [],
						resume_training = False,
						load_model_from = model_path,
						config = conf,
						prediction  = True
						)

	sequence_path, gt_path = get_test_data_path(test_data, gt_available=False)

	for i in xrange(len(sequence_path)):
		print("\n-----------------------------------------------------------------------")
		save_dir = 'path/to/save/prediction/'

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		print("Working on Patient Number", str(i), save_dir)
		print("-----------------------------------------------------------------------\n")

		dice = trainer.LoadandPredict(sequence_path[i], None,
								preProcessList = ['normalize'],
								postProcessList = ['glcc'],
								crf = None,
								save_path = save_dir
								)
		# dices.append(dice)

	# print("\nAverage Dice : " + str(np.mean(dices)))
