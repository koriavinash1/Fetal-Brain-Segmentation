from __future__ import division
import numpy as np
import os, sys, shutil
import tensorflow as tf

# sys.path.insert(0,'../helpers/')
from train_utils import *
# from layers import *

class UNET(object):
	def __init__(self, 
		inputs, 
		targets, 
		weight_maps,
		num_class,  
		n_pool = 3, 
		n_feat_first_layer = [32], 
		chief_class = 1,
		weight_decay = 5e-6, 
		metrics_list = ['dice_loss', 'dice_score'],
		metric_to_optimize = 'dice_loss',
		optimizer = Adam(1e-4),
		gpu_ids =[1,2]):

		self.inputs = inputs
		self.targets = targets
		self.weight_maps = weight_maps
		self.n_pool = n_pool
		self.n_feat_first_layer = n_feat_first_layer
		self.chief_class = chief_class
		self.weight_decay = weight_decay
		self.metrics_list = metrics_list
		self.optimizer = optimizer
		self.gpu_ids = gpu_ids
		self.num_gpus = len(self.gpu_ids)
		self.metric_to_optimize = metric_to_optimize

		self.is_training = tf.placeholder(tf.bool)
		self.num_channels = inputs.get_shape()[-1].value
		self.num_classes = num_class
		one_hot_annotations = tf.one_hot(self.targets, depth=self.num_classes, axis=3)

		self.image_splits = tf.split(self.inputs, self.num_gpus,0)
		self.labels_splits = tf.split(one_hot_annotations, self.num_gpus,0)
		self.target_splits = tf.split(self.targets, self.num_gpus, 0)
		self.weight_splits = tf.split(self.weight_maps, self.num_gpus,0)

		self.logits = {}
		self.posteriors = {}
		self.predictions = {}
		self.grads_dict = {}

		self.stats_ops = {}
		self.inference_ops = {}
		self.accumulate_ops = {}
		self.reset_ops = {}

		for g in gpu_ids:
			g = gpu_ids.index(g)
			self.stats_ops[g] = {}
			self.inference_ops[g] = {}
			self.accumulate_ops[g] = []
			self.reset_ops[g] = []

		with tf.variable_scope(tf.get_variable_scope()) as vscope:
			for i in gpu_ids:
				idx = gpu_ids.index(i)
				with tf.name_scope('tower_%d'%idx):
					with tf.device('/gpu:%d'%i):
						self.doForward(idx,
							n_feat_first_layer = self.n_feat_first_layer,
							n_pool = self.n_pool,
							)
						self.calMetrics(idx)
						tf.get_variable_scope().reuse_variables()

		self.optimize()
		self.makeSummaries()
		self.count_variables()


	def doForward(self,idx, n_feat_first_layer, n_pool):

		inputs = self.image_splits[idx]
		targets = tf.cast(self.target_splits[idx], dtype=tf.float32)
		tf.summary.image('inputs',inputs, max_outputs = 4, collections = ['per_100_steps'])
		tf.summary.image('ground_truth',targets[:,:,:,None], max_outputs = 4, collections = ['per_100_steps'])
		skip_connection_list = []

		#######################
		#   Downsampling path   #
		#######################
		l = inputs
		for i in range(n_pool):
			with tf.variable_scope('Downsampling_Path' + str(i)):
				with tf.variable_scope(str(i)+'Downsampling_Path_Conv_1'):
					l = ReLU(BatchNorm(Conv2D(l, [3,3,l.get_shape()[-1].value, n_feat_first_layer[0]*2**i], 
						collection_name = 'Conv_1' + str(i), padding='SAME'), is_training=self.is_training, isEnabled=False))
				with tf.variable_scope(str(i)+'Downsampling_Path_Conv_2'):
					l = ReLU(BatchNorm(Conv2D(l, [3,3,l.get_shape()[-1].value, n_feat_first_layer[0]*2**i], 
						collection_name = 'Conv_2' + str(i), padding='SAME'), is_training=self.is_training, isEnabled=False))
			print("Downsampling_Path:", i, " shape ", l.get_shape().as_list())
			skip_connection_list.append(l)
			l = MaxPool2(l)
			print("Downsampling_Path_After_Maxpooling:", i, " shape ", l.get_shape().as_list())

		skip_connection_list = skip_connection_list[::-1]

		with tf.variable_scope('Bottleneck_Path'):
			l = ReLU(Conv2D(l, [3,3,l.get_shape()[-1].value, n_feat_first_layer[0]*2**n_pool], 
				collection_name = 'Conv', padding='SAME'))
		print("Bottleneck_Path:", " shape ", l.get_shape().as_list())

		#######################
		#   Upsampling path   #
		#######################

		for i in range(n_pool):
			with tf.variable_scope('Upsampling_Path' + str(i)):
				with tf.variable_scope('Upsampling_Path_Transpose_Conv' + str(i)):
					l = TransposeConv2D(l, n_feat_first_layer[0]*2**(n_pool-i-1), collection_name = 'Transpose_Conv'+str(i))
				print("Upsampling_Path_After_Transpose_Convolution:", i, " shape ", l.get_shape().as_list())
				# Concat skip connection
				l = tf.concat([l, tf.image.resize_image_with_crop_or_pad(skip_connection_list[i],  tf.shape(l)[1],  tf.shape(l)[2])], 3)
				print("Upsampling_Path_After_Concat_Skip:", i, " shape ", l.get_shape().as_list())
				with tf.variable_scope(str(i)+'Upsampling_Path_Conv_1'):
					l = ReLU(BatchNorm(Conv2D(l, [3,3, l.get_shape()[-1].value, n_feat_first_layer[0]*2**(n_pool-i-1)], 
						collection_name = 'Conv_1' + str(i), padding='SAME'), is_training=self.is_training))
				with tf.variable_scope(str(i)+'UPsampling_Path_Conv_2'):
					l = ReLU(BatchNorm(Conv2D(l, [3,3, l.get_shape()[-1].value, n_feat_first_layer[0]*2**(n_pool-i-1)], 
						collection_name = 'Conv_2' + str(i), padding='SAME'), is_training=self.is_training))
			print("Upsampling_Path:", i, " shape ", l.get_shape().as_list())
		#####################
		# 		Outputs 	#
		#####################
		with tf.variable_scope('logits'):
			self.logits[idx] = Conv2D(l, [1,1,l.get_shape()[-1].value,self.num_classes], collection_name = 'logits_layer', padding='SAME')
			print self.logits[idx]

		with tf.variable_scope('logits'):
			self.posteriors[idx] = Softmax(self.logits[idx])	
		print("Final Softmax Layer:", " shape ", self.posteriors[idx].get_shape().as_list())
		with tf.variable_scope('predictions'):
			self.predictions[idx] = tf.cast(tf.argmax(self.logits[idx],3), tf.float32)
			tf.summary.image('predictions',self.predictions[idx][:,:,:,None], max_outputs = 4, collections = ['per_100_steps'])


	def calDiceLoss(self, gpu_id_idx, equal_weight=False):
		# #### Uncomment to enable dice loss
		dice_distance = tf.subtract(1.0, dice_multiclass(self.posteriors[gpu_id_idx], self.labels_splits[gpu_id_idx]))
		class_weights = tf.constant([1.0]*self.num_classes, dtype=tf.float32)	
		dice_loss = tf.divide(tf.reduce_sum(class_weights* dice_distance), tf.reduce_sum(class_weights))
		return dice_loss			

	def calMetrics(self,gpu_id_idx):
		for metric_name in self.metrics_list:
			metric_implemented = False
			if metric_name == 'X_loss':
				metric_implemented = True
				with tf.variable_scope(metric_name):
					# self.inference_ops[gpu_id_idx][metric_name] = loss = SpatialCrossEntropyLogits(self.logits[gpu_id_idx], self.labels_splits[gpu_id_idx]) 
					self.inference_ops[gpu_id_idx][metric_name] = X_loss = \
						 SpatialWeightedCrossEntropyLogits(self.logits[gpu_id_idx], self.labels_splits[gpu_id_idx], self.weight_splits[gpu_id_idx]) 
					#class weights should be in float
					metric_obj = ScalarMetricStream(X_loss)

					tf.summary.scalar('per-step',metric_obj.op,collections = ['per_step'])
					tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])	

			elif metric_name == 'plain_dice_loss':
				metric_implemented = True
				with tf.variable_scope(metric_name):
					self.inference_ops[gpu_id_idx][metric_name] = dice_loss = self.calDiceLoss(gpu_id_idx, equal_weight=True)
					metric_obj = ScalarMetricStream(dice_loss)
					tf.summary.scalar('per-step',metric_obj.op,collections = ['per_step'])
					tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])

			elif metric_name == 'dice_score':
				metric_implemented = True
				with tf.variable_scope(metric_name):
					self.inference_ops[gpu_id_idx][metric_name] = dice_score = \
						DiceCriteria2Cls(self.logits[gpu_id_idx],self.labels_splits[gpu_id_idx],chief_class = self.chief_class)
					metric_obj = ScalarMetricStream(dice_score)
					tf.summary.scalar('per-step',metric_obj.op,collections = ['per_step'])
					tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])

			elif metric_name == 'dice_score_class_0':
				metric_implemented = True
				with tf.variable_scope(metric_name):
					self.inference_ops[gpu_id_idx][metric_name] = dice_score = \
						DiceCriteria2Cls(self.logits[gpu_id_idx],self.labels_splits[gpu_id_idx], chief_class=0)
					metric_obj = ScalarMetricStream(dice_score)
					tf.summary.scalar('per-step',metric_obj.op,collections = ['per_step'])
					tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])

			elif metric_name == 'dice_score_class_1':
				metric_implemented = True
				with tf.variable_scope(metric_name):
					self.inference_ops[gpu_id_idx][metric_name] = dice_score = \
						DiceCriteria2Cls(self.logits[gpu_id_idx],self.labels_splits[gpu_id_idx], chief_class=1)
					metric_obj = ScalarMetricStream(dice_score)
					tf.summary.scalar('per-step',metric_obj.op,collections = ['per_step'])
					tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])				

			elif metric_name == 'dice_loss':
				metric_implemented = True
				with tf.variable_scope(metric_name):
					self.inference_ops[gpu_id_idx][metric_name] = dice_loss = self.calDiceLoss(gpu_id_idx)
					metric_obj = ScalarMetricStream(dice_loss)
					tf.summary.scalar('per-step',metric_obj.op,collections = ['per_step'])
					tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])

			else:
				print('Error : ' + metric_name + ' is not implemented')
			
			if metric_implemented == True: 
				try:		
					self.accumulate_ops[gpu_id_idx].append(metric_obj.accumulate)
					self.stats_ops[gpu_id_idx][metric_name] = metric_obj.stats
					self.reset_ops[gpu_id_idx].append(metric_obj.reset)
				except AttributeError:
					pass

		l2_norm  = sum([tf.nn.l2_loss(v) for v in tf.get_collection('l2_norm_vars')])
		total_loss = self.inference_ops[gpu_id_idx][self.metric_to_optimize] + self.weight_decay*l2_norm

		with tf.variable_scope('gradients'):
			grads = self.optimizer.compute_gradients(total_loss)
		
		self.grads_dict[gpu_id_idx] = grads
		grads = None


	def _averageGradients(self,grads_list):
		average_grads = []
		for grad_and_vars in zip(*grads_list):
			grads = []
			for g, _ in grad_and_vars:
				expanded_g = tf.expand_dims(g, 0)
				grads.append(expanded_g)
				expanded_g = None

			grad = tf.concat(grads, 0)
			grad = tf.reduce_mean(grad, 0)

			v = grad_and_vars[0][1]
			grad_and_var = (grad, v)
			average_grads.append(grad_and_var)
			grad_and_var = None
		return average_grads


	def optimize(self):
		with tf.variable_scope('average_gradients'):
			grads = self._averageGradients(self.grads_dict.values())

		with tf.variable_scope('update_op'):
			self.update_op = self.optimizer.apply_gradients(grads)

	def makeSummaries(self):
		self.summary_ops = {}		
		self.summary_ops['1step'] = tf.summary.merge_all(key = 'per_step')
		self.summary_ops['100steps'] = tf.summary.merge_all(key = 'per_100_steps')	
		self.summary_ops['1epoch'] = tf.summary.merge_all(key = 'per_epoch')

	def count_variables(self):    
		total_parameters = 0
		#iterating over all variables
		for variable in tf.trainable_variables():  
			local_parameters=1
			shape = variable.get_shape()  #getting shape of a variable
			for i in shape:
				local_parameters*=i.value  #mutiplying dimension values
			total_parameters+=local_parameters
		print('Total Number of Trainable Parameters:', total_parameters) 