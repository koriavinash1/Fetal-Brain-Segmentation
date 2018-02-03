import tensorflow as tf
import numpy as np
import sys

def flexiSession():
	config = tf.ConfigProto()
	config.allow_soft_placement = True
	config.gpu_options.allow_growth = True
	return tf.Session(config = config)

def getWeightAndBias(weights_shape, bias_shape, collection_name, non_zero_bias=True):
	"""
	TODO: Check Change initializer
	"""
	initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
	W = tf.get_variable(shape = weights_shape, name = 'weight_matrix',
		initializer = initializer)
	tf.add_to_collection('l2_norm_variables',W)
	tf.add_to_collection(collection_name,W)

	if non_zero_bias:
		bias_initializer = tf.zeros_initializer()
		b = tf.get_variable(name = 'biases', shape = bias_shape, initializer = bias_initializer)
		tf.add_to_collection('l2_norm_variables',b)
		tf.add_to_collection(collection_name,b)
	else:
		b = tf.constant(0.0, name = 'constant_zero_biases', shape = bias_shape)
	return W, b

def getInputsPlaceholder(*shape):
	return tf.placeholder(tf.float32, shape, name='Inputs')

def getTargetsPlaceholder(*shape):
	return tf.placeholder(tf.uint8, shape, name='Targets')

def OneHot(targets,num_class):
  return tf.one_hot(targets,num_class,1,0)

def Softmax(logits):
	return tf.nn.softmax(logits,name = 'softmax')

def Dropout(inputs, is_training, keep_prob = 0.7):
	keep_prob_pl = tf.cond(is_training, lambda : tf.constant(keep_prob), lambda : tf.constant(1.0))
	return tf.nn.dropout(inputs,keep_prob_pl)

def Conv2D(inputs, weights_shape, collection_name = '', stride = 1, padding = 'VALID', non_zero_bias=True):
	"""
	### TO-DO ###
	If we want some specific padding value
	"""
	strides = [1,stride,stride,1]

	W_shape = weights_shape
	b_shape = [weights_shape[3]]
	W, b = getWeightAndBias(W_shape, b_shape,collection_name = collection_name, non_zero_bias=non_zero_bias)
	output = tf.nn.conv2d(inputs, W, strides, padding)
	output = tf.add(output, b, name ='add_bias')
	return output

def TransposeConv2D(inputs, n_filters_keep, collection_name = '', filter_size = (3,3), stride = (2,2), padding = 'SAME', non_zero_bias=True):
	"""
	### TO-DO ###
	"""
	def deconv_output_length(input_length, filter_size, padding, stride):
		if input_length is None:
			return None

		input_length *= stride
		if padding == 'VALID':
			input_length += max(filter_size - stride, 0)
		elif padding == 'FULL':
			input_length -= (stride + filter_size - 2)
		return input_length

	input_shape = tf.shape(inputs)
	batch_size, height, width = input_shape[0], input_shape[1], input_shape[2]
	kernel_h, kernel_w = filter_size
	stride_h, stride_w = stride

	# Infer the dynamic output shape:
	out_height = deconv_output_length(height,
					kernel_h,
					padding,
					stride_h)
	out_width = deconv_output_length(width,
					kernel_w,
					padding,
					stride_w)

	output_shape = (batch_size, out_height, out_width, n_filters_keep)
	output_shape_tensor = tf.stack(output_shape)
	strides = [1, stride_h, stride_w, 1]
	W_shape = [kernel_h, kernel_w, n_filters_keep, inputs.get_shape()[-1].value]
	b_shape = [n_filters_keep]
	W, b = getWeightAndBias(W_shape, b_shape, collection_name = collection_name, non_zero_bias=non_zero_bias)
	output = tf.nn.conv2d_transpose (inputs, W, output_shape_tensor, strides, padding=padding)
	output = tf.add(output, b, name ='add_bias')
	return output

def Elu(x):
	return tf.nn.elu(x)

def ReLU(x):
	return tf.nn.relu(x)

def MaxPool2(x):
	output = tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID')
	return output

def BatchNorm(inputs, is_training, decay = 0.9, epsilon=1e-3, isEnabled=True):
	# TODO: Check the effect of batch_norm is True Always
	# It is observed that batch norm affects the quality of segmentation results
	if not isEnabled:
		return inputs
	# is_training=tf.constant(True, dtype=tf.bool)
	
	with tf.device('/cpu:0'):
		scale = tf.get_variable(name = 'scale', shape = inputs.get_shape()[-1],
			initializer = tf.constant_initializer(1.0),dtype = tf.float32)
		tf.add_to_collection('l2_norm_variables', scale)
		beta = tf.get_variable(name = 'beta', shape = inputs.get_shape()[-1],
			initializer = tf.constant_initializer(0.0),dtype = tf.float32)
		tf.add_to_collection('l2_norm_variables',beta)
	pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
	pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
	axis = list(range(len(inputs.get_shape())-1))

	def Train(inputs, pop_mean, pop_var, scale, beta):
		batch_mean, batch_var = tf.nn.moments(inputs,axis)
		train_mean = tf.assign(pop_mean,
							   pop_mean * decay + batch_mean * (1 - decay))
		train_var = tf.assign(pop_var,
							  pop_var * decay + batch_var * (1 - decay))

		mean_distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pop_mean, batch_mean))))
		var_distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pop_var, batch_var))))

		with tf.control_dependencies([train_mean, train_var]):
			return tf.nn.batch_normalization(inputs,
				batch_mean, batch_var, beta, scale, epsilon)

	def Eval(inputs, pop_mean, pop_var, scale, beta):
		return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

	return tf.cond(is_training, lambda: Train(inputs, pop_mean, pop_var, scale, beta),
		lambda: Eval(inputs, pop_mean, pop_var, scale, beta))

def ConvEluBatchNormDropout(inputs, shape, stride = 1, padding = 'VALID', bn_mode = tf.placeholder_with_default(False, shape = []), drop_mode = tf.placeholder_with_default(False, shape = []), keep_prob = 0.7, collections = []):
	return Dropout(BatchNorm(ConvElu(inputs,shape,stride,padding, collections = collections),bn_mode, collections = collections), drop_mode,keep_prob)

def SpatialBilinearUpsampling(x,factor = 2):
	shape = [tf.shape(x)[1]*factor,tf.shape(x)[2]*factor]
	return tf.image.resize_bilinear(x,shape)

def TransitionDown(inputs, n_filters,collection_name, keep_prob=0.8, is_training=tf.constant(False,dtype=tf.bool)):
	""" Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """
	l = BN_eLU_Conv(inputs, n_filters,collection_name=collection_name, filter_size=1, keep_prob=keep_prob, is_training=is_training)
	l = MaxPool2(l)

	return l

def BN_eLU_Conv(inputs, n_filters,collection_name, filter_size=3, keep_prob=0.8, is_training=tf.constant(False,dtype=tf.bool), drop_BN=False, use_elu=True):
	l = inputs
	if not drop_BN:
		l = BatchNorm(l, is_training=is_training)
	if use_elu:
		l = Elu(l)
	else:
		l = ReLU(l)        
	l = Conv2D(l, [filter_size, filter_size, l.get_shape()[-1].value, n_filters], collection_name = collection_name, padding='SAME')
	l = Dropout(l, is_training=is_training,keep_prob=keep_prob)
	return l

def dice_multiclass(output, target, loss_type='sorensen', axis=[0,1,2], smooth=1e-5):
	inse = tf.reduce_sum(output * target, axis=axis)
	if loss_type == 'jaccard':
		l = tf.reduce_sum(output * output, axis=axis)
		r = tf.reduce_sum(target * target, axis=axis)
	elif loss_type == 'sorensen':
		l = tf.reduce_sum(output, axis=axis)
		r = tf.reduce_sum(target, axis=axis)
	else:
		raise Exception("Unknow loss_type")
	# dice = 2 * (inse) / (l + r)
	# epsilon = 1e-5
	# dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1

	dice = (2. * inse + smooth) / (l + r + smooth)
	##Attention: Return dice/jaccard score of all the classes in the batch if axis=0
	# dice = tf.reduce_mean(dice, axis=0)
	return dice

def Adam(lr):
	return tf.train.AdamOptimizer(learning_rate = lr)

def progress(curr_idx, max_idx, time_step,repeat_elem = "_"):
	max_equals = 55
	step_ms = int(time_step*1000)
	num_equals = int(curr_idx*max_equals/float(max_idx))
	len_reverse =len('Step:%d ms| %d/%d ['%(step_ms, curr_idx, max_idx)) + num_equals
	sys.stdout.write("Step:%d ms|%d/%d [%s]" %(step_ms, curr_idx, max_idx, " " * max_equals,))
	sys.stdout.flush()
	sys.stdout.write("\b" * (max_equals+1))
	sys.stdout.write(repeat_elem * num_equals)
	sys.stdout.write("\b"*len_reverse)
	sys.stdout.flush()
	if curr_idx == max_idx:
		print('\n')


def SpatialWeightedCrossEntropyLogits(logits, targets, weight_map, name='spatial_wx_entropy'):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = targets,logits = logits)
	weighted_cross_entropy = tf.multiply(cross_entropy, weight_map)
	mean_weighted_cross_entropy = tf.reduce_mean(weighted_cross_entropy, name=name)
	return mean_weighted_cross_entropy

def SpatialCrossEntropyLogits(logits,targets, name='spatial_cross_entropy'):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = targets,logits = logits)
	return tf.reduce_mean(cross_entropy, name=name)

def DiceCriteria2Cls(logits, targets, chief_class, smooth = 1.0, name = 'dice_score'):
	last_dim_idx = logits.get_shape().ndims - 1
	num_class = tf.shape(logits)[last_dim_idx]
	predictions = tf.one_hot(tf.argmax(logits,last_dim_idx),num_class)
	preds_unrolled = tf.reshape(predictions,[-1,num_class])[:,chief_class]
	targets_unrolled = tf.reshape(targets,[-1,num_class])[:,chief_class]
	intersection = tf.reduce_sum(preds_unrolled*targets_unrolled)
	ret_val = (2.0*intersection)/(tf.reduce_sum(preds_unrolled)
	 + tf.reduce_sum(targets_unrolled) + smooth)
	ret_val = tf.identity(ret_val,name = 'dice_score')
	return ret_val

class ScalarMetricStream(object):
	def __init__(self,op, filter_nan = False):
		self.op = op
		count = tf.constant(1.0)
		self.sum = tf.Variable([0.0,0], name = op.name[:-2] + '_sum', trainable = False)
		self.avg = tf.Variable(0.0, name = op.name[:-2] + '_avg', trainable = False)

		if filter_nan == True:
			op_is_nan = tf.is_nan(self.op)
			count = tf.cond(op_is_nan, lambda : tf.constant(0.0), lambda : tf.constant(1.0))
			self.op = tf.cond(op_is_nan, lambda : tf.constant(0.0),lambda : tf.identity(self.op))
		self.accumulate = tf.assign_add(self.sum,[self.op,count])
		self.reset = tf.assign(self.sum,[0.0,0.0])
		self.stats = tf.assign(self.avg,self.sum[0]/(0.001 + self.sum[1]))