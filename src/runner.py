import argparse
import os, sys, shutil
import numpy as np 
import time, sys
from train_utils import *
from estimator import *
from config import *
from networks import *

if __name__ == '__main__':
	conf = conf()
	run_dir = os.path.join(conf.output_dir,conf.run_name)
	model_path = os.path.join(run_dir,'models','latest.ckpt')

	if conf.resume_training and (conf.load_model_from == None):
		conf.load_model_from = model_path

	summary_dir = os.path.join(conf.output_dir,conf.run_name,'summary')
	# freq_list = ['per_step', 'per_25_steps', 'per_100_steps','per_epoch','per_five_epochs']
	freq_list = ['per_step', 'per_100_steps', 'per_epoch']


	inputs = getInputsPlaceholder(None, None, None, conf.num_channels)
	targets = getTargetsPlaceholder(None, None, None)
	weight_maps = tf.placeholder(tf.float32, shape=[None, None, None])

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
				summary_dir = summary_dir,
				freq_list = freq_list,
				resume_training = conf.resume_training,
				load_model_from = conf.load_model_from,
				config = conf
				)
	# iterate for the number of epochs
	for epoch in range(int(trainer.numKeeper.counts['epoch']+1), conf.num_epochs):
		print('\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
		print('Training @ Epoch : ' + str(epoch))
		trainer.fit(steps=-1)
		print('\n---------------------------------------------------')
		print('Validating @ Epoch : ' + str(epoch))
		trainer.evaluate(steps=-1 )
		trainer.numKeeper.counts['epoch'] = trainer.numKeeper.counts['epoch'] + 1
		trainer.numKeeper.updateCounts(trainer.summary_manager.counts)
		trainer.saveModel(os.path.join(conf.output_dir,conf.run_name,'models','latest.ckpt'))
