from __future__ import division
import tensorflow as tf 
import numpy as np 
import os, sys
import pprint
import time, argparse
import collections
import cv2, sys
from data_loader import ITERATOR
from train_utils import *
import matplotlib.patches as patches

class Estimator(object):
	def __init__(self,
		 net_obj,
		 summary_dir,
		 freq_list,
		 resume_training,
		 load_model_from,
		 config,
		 prediction = False):

		self.conf = config
		self.net = net_obj
		self.prediction = prediction

		if not self.prediction:
			print("batch size : " + str(self.conf.batch_size))
			print("Initialising data loader ...")		
			self.data_iterator = ITERATOR(self.conf.data_folder, mode = 'train',	batch_size = self.conf.batch_size, num_threads=1)

			print('preparing summary manager')
			self.summary_manager = SummaryManager(summary_dir,tf.get_default_graph(),freq_list)

			print('Initialising the numbers manager')
			self.numKeeper = numbersKeeper()
		
			#initalising numbersKeeper
			counts_initializer = {'epoch' : 0,'dice_score' : 0}
			for freq in freq_list:
				counts_initializer['train'] = self.summary_manager.counts['train']
				counts_initializer['valid'] = self.summary_manager.counts['valid']
				counts_initializer['test'] = self.summary_manager.counts['test']

			self.numKeeper.initNumpyDict(counts_initializer)
			
			print('defining the session')
			self.sess = flexiSession()
			self.sess.run(tf.global_variables_initializer())
			
			try:
				self.sess.run(tf.assert_variables_initialized())
			except tf.errors.FailedPreconditionError:
				raise RuntimeError('Not all variables initialized')

			self.saver = tf.train.Saver(tf.global_variables())
		
			if ((resume_training == True) and (load_model_from is not None)):
				self.restoreModel(load_model_from)

			if ((resume_training == False) and (load_model_from is not None)):
				self.restoreOnlyModel(load_model_from)			

		else:
			self.binary_opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
			self.binary_opening_filter.SetKernelRadius(1)

			self.binary_closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
			self.binary_closing_filter.SetKernelRadius(1)

			self.erosion_filter = sitk.BinaryErodeImageFilter()
			self.erosion_filter.SetKernelRadius(1)

			self.dilation_filter = sitk.BinaryDilateImageFilter()
			self.dilation_filter.SetKernelRadius(1)

			print('defining the session')
			self.sess = utils.flexiSession()
			self.sess.run(tf.global_variables_initializer())

			self.saver = tf.train.Saver(tf.global_variables())
			self.restoreOnlyModel(load_model_from)

	def restoreOnlyModel(self, load_model_from):
		print('Restoring model from: ' + str(load_model_from))
		self.saver.restore(self.sess,load_model_from)

	def restoreModel(self,load_model_from):
		print('Restoring model from: ' + str(load_model_from))
		self.saver.restore(self.sess,load_model_from)
		self.numKeeper.updateCounts(self.sess.run(self.numKeeper.tf_counts))
		print("\nWhile restoring : ")
		print(self.sess.run(self.numKeeper.tf_counts))
		print("\n")

		self.summary_manager.update_counts(self.numKeeper.counts)
		print('Epochs completed : ' + str(self.numKeeper.counts['epoch']))
		print('Best dice: ' + str(self.numKeeper.counts['dice_score']))

	def saveModel(self,save_path):
		self.sess.run(self.numKeeper.assignNpToTfVariables(self.numKeeper.counts))
		model_dir = os.path.split(save_path)[0]
		if not os.path.isdir(model_dir): 
			os.makedirs(model_dir)
		self.saver.save(self.sess, save_path)


	def fit(self, steps=1000):
		self.data_iterator.mode = 'train'
		self.summary_manager.mode = 'train'
		self.data_iterator.reset()
		time.sleep(5.0)

		feed = None
		train_ops = [self.net.inference_ops,
				self.net.summary_ops['1step'],
				self.net.update_op,
				self.net.accumulate_ops,
				self.net.logits]
		count = 0
		step_sec = 0
		if steps <= 0:
			steps = self.data_iterator.train_steps
		while (count < steps):
			start_time = time.time()

			# fetch inputs batches and verify if they are numpy.ndarray and run all the ops
			# g_time = time.time()
			input_batch, target_batch, weight_batch = self.data_iterator.getNextBatch()
			# print("time taken to get a batch : " + str(time.time()-g_time) + 's')

			if type(input_batch) is np.ndarray:

				feed = {self.net.inputs : input_batch,
						self.net.targets : target_batch,
						self.net.weight_maps : weight_batch,
						self.net.is_training : True}
				input_batch, target_batch, weight_batch = None, None, None
				# i_time = time.time()
				inferences, summary, _, __, outputs = self.sess.run(train_ops,feed_dict =  feed)
				# print("time taken to for inference: " + str(time.time()-i_time) + 's')
				# print("\n")

				self.summary_manager.addSummary(summary,"train","per_step")
				progress(count%steps+1,steps, step_sec)
				
				inferences, outputs, summary = None, None, None

				# add summaries regularly for every 100 steps
				if (count + 1)% 25 == 0:
					summary = self.sess.run(self.net.summary_ops['100steps'], feed_dict = feed)	
					self.summary_manager.addSummary(summary,"train","per_100_steps")

				if (count + 1) % 250 == 0:
					print('Avg metrics : ')
					pprint.pprint(self.sess.run(self.net.stats_ops), width = 1)
				count = count + 1 

			stop_time = time.time()
			step_sec = stop_time - start_time
			if self.data_iterator.iter_over == True:
				self.data_iterator.reset()
				time.sleep(4)


		print('\nAvg metrics for epoch : ')
		pprint.pprint(self.sess.run(self.net.stats_ops), width = 1)
		summary = self.sess.run(self.net.summary_ops['1epoch'],
			feed_dict = feed)
		self.sess.run(self.net.reset_ops)
		self.summary_manager.addSummary(summary,"train","per_epoch")
		summary = None


	def evaluate(self, steps=250):
		#set mode and wait for the threads to populate the queue
		self.data_iterator.mode = 'valid'
		self.summary_manager.mode = 'valid'
		self.data_iterator.reset()
		time.sleep(5.0)
		
		feed = None
		valid_ops = [self.net.inference_ops,
					 self.net.summary_ops['1step'],
					 self.net.accumulate_ops,
					 self.net.logits]

		#iterate the validation step, until count = steps
		count = 0
		step_sec = 0
		if steps <= 0:
			steps = self.data_iterator.valid_steps
		while (count < steps):
			start_time = time.time()
			time.sleep(0.4)

			input_batch, target_batch, weight_batch = self.data_iterator.getNextBatch()
			if type(input_batch) is np.ndarray:

				feed = {self.net.inputs : input_batch,
						self.net.targets : target_batch,
						self.net.weight_maps :  weight_batch,
						self.net.is_training : False}

				input_batch, target_batch, weight_batch = None, None, None
				
				inferences, summary, _, outputs = self.sess.run(valid_ops, feed_dict = feed) 
				self.summary_manager.addSummary(summary,"valid", "per_step")
				progress(count%steps+1,steps, step_sec)
				inferences, outputs, summary = None, None, None
				
				if (count+1) % 5 == 0:
					summary = self.sess.run(self.net.summary_ops['100steps'], feed_dict = feed)
					self.summary_manager.addSummary(summary,"valid", "per_100_steps")

				if (count+1) % 250 == 0:
					print('Avg metrics : ')
					pprint.pprint(self.sess.run(self.net.stats_ops), width = 1)

				count = count + 1 
			stop_time = time.time()
			step_sec = stop_time - start_time
			if self.data_iterator.iter_over == True:
				print('\nIteration over')
				self.data_iterator.reset()
				time.sleep(5)

		print('\nAvg metrics for epoch : ')
		metrics = self.sess.run(self.net.stats_ops)
		pprint.pprint(metrics, width=1)
		if (metrics[0]['dice_score'] > self.numKeeper.counts['dice_score']):
			self.numKeeper.counts['dice_score'] = metrics[0]['dice_score']
			self.numKeeper.updateCounts(self.summary_manager.counts)
			print('Saving best model!')
			self.saveModel(os.path.join(self.conf.output_dir,self.conf.run_name,'best_model','latest.ckpt'))

		print('Current best dice: ' + str(self.numKeeper.counts['dice_score']))
		summary = self.sess.run(self.net.summary_ops['1epoch'],
			feed_dict = feed)
		self.sess.run(self.net.reset_ops)
		self.summary_manager.addSummary(summary,"valid","per_epoch")
		summary = None

	def CalDice(self, pred, gt, class_lbl=1):
		"""
		Calculate dice score
		"""
		# intersection = np.sum((pred == class_lbl) & (gt == class_lbl))
		# dice_4d = 2.0 *(intersection)/(np.sum(pred == class_lbl) + np.sum(gt == class_lbl))
		dices = []
		labelPred=sitk.GetImageFromArray(pred, isVector=False)
		labelTrue=sitk.GetImageFromArray(gt, isVector=False)
		dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
		dicecomputer.Execute(labelTrue==class_lbl,labelPred==class_lbl)
		dice=dicecomputer.GetDiceCoefficient()
		dices.append(dice)
		# print (np.mean(dices), dice_4d)
		return np.mean(dices)

	def Predict(self, input_vol_list, postProcessList = [], crf = None, roi_mask_path = None):
		"""
		Infer phase-wise- 3D batches
		"""
		x_dim = input_vol_list[0].shape[0]
		y_dim = input_vol_list[0].shape[1]
		n_class = self.conf.num_class
		n_channel = self.conf.num_channels
		n_batch = self.conf.prediction_batch_size
		# Intialization
		s_time = time.time()
		output_volume = np.zeros([x_dim, y_dim])
		out_posteriors = np.zeros([x_dim, y_dim, n_class])
		img_batch = np.zeros([n_batch, x_dim, y_dim, n_channel])
		for i in range(n_batch):
			img_batch[i,:,:,:] = input_vol_list[0]
			outputs, posteriors = self.sess.run([self.net.predictions, self.net.posteriors], 
				feed_dict ={self.net.inputs: img_batch, self.net.is_training:False})
			output_volume = outputs[0][0]
			out_posteriors = posteriors[0]
			print (outputs[0][0, :,:].shape, posteriors[0].shape, posteriors[0].shape)
		imshow(img_batch[0,:,:,:], outputs[0][0, :,:], posteriors[0][0,:,:,0], posteriors[0][0,:,:,1])

		print (output_volume.shape)
		output_volume = self.PostprocessVolume(output_volume, postProcessList, roi_mask_path)
		print (output_volume.shape)
		# TODO: CRF
		if crf is not None:
			output_volume = doCRF(output_volume, out_posteriors)

		# Resize the output 3d volume to its original dimension 
		output_volume = self.PostUpsample(output_volume, postProcessList)

		return output_volume

	def SavePrediction(self, vol_pred, prefix, save_path):
		"""
		TODO:
		"""
		print (vol_pred.shape)
		cv2.imwrite(os.path.join(save_path, prefix), vol_pred)
		return
		
	def LoadAndPredict(self, img_files_path_list, seg_files_path_list = None, roi_mask_path = None, 
						patch_size=(128, 128), outputs_shape = None, preProcessList=[], 
						postProcessList = [], crf=None, save_path = None):
		"""
		TODO: Generic Prediction pipeline 
		"""

		# print("loading image and pre-processing ...")
		patient_data = read_img(img_files_path_list, seg_files_path_list, size=512)
		vol_data = [patient_data['img']]

		vol_data = self.LoadandPreprocess(vol_data, preProcessList)
		vol_pred = self.Predict(vol_data, postProcessList, crf, roi_mask_path)
		# print (pred_3d.shape)
		if seg_files_path_list:
			print("loading segmentation and computing Quality metrics...")
			vol_gt = patient_data['gt']
			dice = self.CalDice(vol_pred, vol_gt, class_lbl=1)
			print("Dice: " + os.path.basename(save_path) + " : " + str(dice))

		if not save_path is None:
			print("Saving Predictions in jpg format ...")
			# Get Affine Matrix
			# TODO: Generalize
			prefix = 'test-segmentation-' + os.path.basename(img_files_path_list)
			self.SavePrediction(vol_pred, prefix, save_path)

		if seg_files_path_list:
			return (dice)
		else:
			return (0, 0)


class SummaryManager(object):
	def __init__(self,summary_dir, model_graph, freq_list, mode='train'):
		self.summary_dir = summary_dir
		self.model_graph = model_graph
		self.mode = mode

		# Create the different directories to save summaries of train, valid , test 
		self.train_dir = self.checkAndCreateFolder(os.path.join(summary_dir,'train'))
		self.valid_dir = self.checkAndCreateFolder(os.path.join(summary_dir,'valid'))
		self.test_dir = self.checkAndCreateFolder(os.path.join(summary_dir,'test'))

		#Create different summary writers for train, valid and test
		self.trainWriter = tf.summary.FileWriter(self.train_dir,model_graph)
		self.validWriter = tf.summary.FileWriter(self.valid_dir,model_graph)
		self.testWriter = tf.summary.FileWriter(self.test_dir,model_graph)

		# Initialise the counts
		self.counts = {'train': {},'valid': {},'test': {}}

		# Initialise the counts
		self.initialiseCounts(freq_list)

	def update_counts(self,count_dict):
		for mode, val1 in count_dict.items():
			if isinstance(val1,collections.Mapping):
				for freq, val in count_dict[mode].items():
					self.counts[mode][freq] = val

	def initialiseCounts(self,freq_list):
		for each in freq_list:
			self.counts['train'][each] = 0
			self.counts['valid'][each] = 0
			self.counts['test'][each] = 0

	def checkAndCreateFolder(self,path):
		if not os.path.exists(path):
			os.makedirs(path)
		return path
	def addSummary(self,summary,mode,time_step_name):
		if self.mode == 'train':
			self.trainWriter.add_summary(summary,self.counts[mode][time_step_name])
			self.counts[mode][time_step_name] += 1
		elif self.mode == 'valid':
			self.validWriter.add_summary(summary,self.counts[mode][time_step_name])
			self.counts[mode][time_step_name] += 1
		elif self.mode == 'test':
			self.testWriter.add_summary(summary,self.counts[mode][time_step_name])
			self.counts[mode][time_step_name] += 1
		else:
			print("summary not added !!!, accepted values for variable mode are 'train', 'valid', 'test' ")
			raise 1


class numbersKeeper(object):
	def __init__(self):
		self.counts = {'train':{},'valid':{},'test':{}}
		self.tf_counts = {'train':{},'valid':{},'test':{}}

	def updateCounts(self, count_dict):
		for mode, val1 in count_dict.items():
			if isinstance(val1,collections.Mapping):
				for freq, val in count_dict[mode].items():
					self.counts[mode][freq] = val
			else:
				self.counts[mode] = val1

	def initNumpyDict(self,count_dict):
		for mode,val1 in count_dict.items():
			if isinstance(val1,collections.Mapping):
				for freq, val in count_dict[mode].items():
					self.counts[mode][freq] = val
			else:
				self.counts[mode] = val1

		self.initTfVariables(count_dict)

	def initTfVariables(self,count_dict):
		# print(count_dict)
		for mode,val1 in count_dict.items():
			if isinstance(val1,collections.Mapping):
				for freq, val in count_dict[mode].items():
					self.tf_counts[mode][freq] = tf.Variable(val,name=mode+freq,trainable=False)
			else:
				self.tf_counts[mode] = tf.Variable(val1,dtype=tf.float32,name=mode,trainable=False)


	def assignNpToTfVariables(self,count_dict):
		assign_ops = []
		for mode,val1 in count_dict.items():
			if isinstance(val1,collections.Mapping):
				for freq, val in count_dict[mode].items():
					assign_ops.append(tf.assign(self.tf_counts[mode][freq],val))
			else:
				assign_ops.append(tf.assign(self.tf_counts[mode],val1))

		return assign_ops