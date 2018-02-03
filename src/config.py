import numpy as np
import argparse
import os

class conf(object):
	def __init__(self,
		data_folder = "../train_test_split.csv",
		output_dir = '../models/',
		# run_name = 'U-NET_Aug_Tx_Ro_Noi_F_DiceLoss',
		run_name = 'U-NET__DiceLoss',
		# run_name = 'U-NET_Aug_Tx_Ro_Noi_F_Z_Df_CELoss',
		# run_name = 'U-NET_Aug_Tx_Ro_Noi_F_Z_Df_XLoss',
		# run_name = 'U-NET_Aug_Tx_Ro_Noi_F_Z_Df_XWDLoss',
		# run_name = 'FIRD_Aug_Tx_Ro_Noi_F_Z_Df_XWDLoss',
		batch_size = 16,
		chief_class = 1,
		num_class = 2,
		num_channels = 1,
		num_epochs = 300,
		num_gpus = 1,
		gpu_ids = [0],
		resume_training = True,
		load_model_from=None,
		prediction = False,
		prediction_batch_size=1,
		learning_rate = 1e-3
		):

		self.data_folder = data_folder
		self.output_dir = output_dir
		self.run_name = run_name
		self.batch_size = batch_size
		self.chief_class = chief_class
		self.num_channels = num_channels
		self.num_class = num_class
		self.num_epochs = num_epochs
		self.num_gpus = num_gpus
		self.gpu_ids = gpu_ids
		self.resume_training = resume_training
		self.load_model_from = load_model_from
		self.prediction = prediction
		self.prediction_batch_size = prediction_batch_size
		self.learning_rate = learning_rate

		# Data Augmentation Parameters
		# Set patch extraction parameters
		size0 = (64, 64)
		size1 = (128, 128)
		size2 = (256, 256)
		patch_size = size1
		max_size = size1
		self.patch_size = patch_size
		# Set Environment
		os.environ['CUDA_VISIBLE_DEVICES'] = '0'