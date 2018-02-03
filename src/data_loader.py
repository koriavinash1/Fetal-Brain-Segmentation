from __future__ import division
import numpy as np
import os, sys, shutil
import h5py
import pandas as pd
import random, time
import scipy.ndimage as snd
import skimage.morphology as morph
import weakref
import threading
import random
import matplotlib.pyplot as plt
from config import *
conf = conf()
def imshow(*args,**kwargs):
	""" Handy function to show multiple plots in on row, possibly with different cmaps and titles
	Usage: 
	imshow(img1, title="myPlot")
	imshow(img1,img2, title=['title1','title2'])
	imshow(img1,img2, cmap='hot')
	imshow(img1,img2,cmap=['gray','Blues']) """
	cmap = kwargs.get('cmap', 'gray')
	title= kwargs.get('title','')
	if len(args)==0:
		raise ValueError("No images given to imshow")
	elif len(args)==1:
		plt.title(title)
		plt.imshow(args[0], interpolation='none')
	else:
		n=len(args)
		if type(cmap)==str:
			cmap = [cmap]*n
		if type(title)==str:
		    	title= [title]*n
		plt.figure(figsize=(n*5,10))
		for i in range(n):
			plt.subplot(1,n,i+1)
			plt.title(title[i])
			plt.imshow(args[i], cmap[i])
	plt.show()
    

def worker(weak_self):
	self = weak_self()
	name  = threading.current_thread().name

	while not self.done_event.is_set():
		if (self.iter_over == False) and (self.n_imgs_in_ram < self.max_imgs_in_ram):
			with self.file_access_lock:
				input_path = self.popFilePath()

				if (input_path is not None) and (input_path not in self.files_accessed):
					self.files_accessed.append(input_path)
					image, label, weight = self.getDataFromPath(input_path)
					# print image.shape, label.shape, weight.shape
					with self.data_access_lock:
						if self.image_volume.size != 0  and self.label_volume.size != 0:
							try:
								self.image_volume = np.vstack([self.image_volume, image])
								self.label_volume = np.vstack([self.label_volume, label])
								self.weight_volume = np.vstack([self.weight_volume, weight])
							except Exception as e:
								print(str(e))
								self.image_volume = np.array([])
								self.label_volume = np.array([])
								self.weight_volume = np.array([])
								print('Image queue shape: ' + str(self.image_volume.shape))
								print('Image slice shape: ' + str(image.shape))
								print('Label queue shape: ' + str(self.label_volume.shape))
								print('Label slice shape: ' + str(label.shape))
						else:
							self.image_volume = image
							self.label_volume = label
							self.weight_volume = weight

						self.n_imgs_in_ram = self.image_volume.shape[0]
				
				elif input_path == None:
					self.iter_over_for_thread[name] = True



class ITERATOR(object):
	def __init__(self, data_folder_path, mode='train', batch_size = 2, num_threads = 4, max_imgs_in_ram = 500):
		self.data_folder_path = data_folder_path
		self.mode = mode
		self.batch_size = batch_size
		self.num_threads = num_threads
		self.iter_over = False
		self.mode = 'train'
		self.image_volume = np.array([])
		self.label_volume = np.array([])
		self.weight_volume = np.array([])
		self.n_imgs_in_ram = 0
		self.num_imgs_obt = 0
		self.max_imgs_in_ram = max_imgs_in_ram
		self.files_accessed = []
		self.file_access_lock = threading.Lock()
		self.data_access_lock = threading.Lock()
		self.done_event = threading.Event()
		self.iter_over_for_thread = {}

		self.getFilePaths(data_folder_path)

		for t_i in range(0,num_threads):
			t = threading.Thread(target = worker,args = (weakref.ref(self),))
			t.setDaemon(True)
			t.start()
			self.iter_over_for_thread[t.name] = False

	def getFilePaths(self,data_folder_path):
		train_valid_paths = pd.read_csv(self.data_folder_path)
		self.train_fls = np.squeeze(train_valid_paths[train_valid_paths['Training']]['Slice Path'].as_matrix()).tolist()
		self.val_fls = np.squeeze(train_valid_paths[train_valid_paths['Validation']]['Slice Path'].as_matrix()).tolist()
		self.test_fls = np.squeeze(train_valid_paths[train_valid_paths['Testing']]['Slice Path'].as_matrix()).tolist()

		random.shuffle(self.test_fls)
		random.shuffle(self.train_fls)
		random.shuffle(self.val_fls)

		self.train_steps = len(self.train_fls)//self.batch_size
		self.valid_steps = len(self.val_fls)//self.batch_size

	def getPatchSize(self, image, label, weight_map, target_size = conf.patch_size):
		shape = image.shape
		x = np.random.randint(0, shape[1] - target_size[0])
		y = np.random.randint(0, shape[2] - target_size[1])
		image_patch = image[:,x:x+target_size[0]:, y:y+target_size[1]:,]
		label_patch = label[:,x:x+target_size[0]:, y:y+target_size[1]]
		weight_map_patch = weight_map[:,x:x+target_size[0]:, y:y+target_size[1]] 
		# print "###########################################"
		# print image_patch.shape, label_patch.shape
		return image_patch, label_patch, weight_map_patch

	def popFilePath(self):

		if self.mode == 'train':
			if len(self.train_fls) > 0:
				return self.train_fls.pop()
			else:
				return None
		elif self.mode == 'valid':
			if len(self.val_fls) > 0:
				return self.val_fls.pop()
			else:
				return None
		else:
			print("Got unknown value for mode, supported values are : 'train', 'val' ")
			raise 1

	def getDataFromPath(self,path):
		h5 = h5py.File(path,'r')
		img = h5['image'][:]
		lbl = h5['label'][:]
		weight = h5['weight_map'][:]
		img, lbl, weight = self.getPatchSize(img, lbl,weight)
		return img, lbl, weight

	def getNextBatch(self):
		temp_count = 0
		while True:
			with self.data_access_lock:
				image_batch, self.image_volume = np.split(self.image_volume, [self.batch_size])
				label_batch,self.label_volume = np.split(self.label_volume,[self.batch_size])
				weight_batch,self.weight_volume = np.split(self.weight_volume,[self.batch_size])

				num_imgs_obt = image_batch.shape[0]
				self.n_imgs_in_ram = self.image_volume.shape[0]

			if ((sum(x == True for x in self.iter_over_for_thread.values()) == self.num_threads) and (self.n_imgs_in_ram == 0)):
				self.iter_over = True

			if (num_imgs_obt > 0) or self.iter_over :
				if (num_imgs_obt != self.batch_size) and temp_count <=3 :
					time.sleep(2)
					temp_count += 1
				else:
					break

		return image_batch, label_batch, weight_batch

	def reset(self):
		self.image_volume = np.array([])
		self.label_volume = np.array([])
		self.weight_volume = np.array([])
		self.n_imgs_in_ram = self.image_volume.shape[0]
		self.train_fls = []
		self.val_fls = []
		self.getFilePaths(self.data_folder_path)
		self.files_accessed = []
		for key in self.iter_over_for_thread:
			self.iter_over_for_thread[key] = False	
		self.iter_over = False

	def __del__(self):
		print(' Thread exited ')
		self.done_event.set()


if __name__ == '__main__':
	# Path to data folder
	data_path = "../train_test_split.csv"
	batch_size = 10
	# Set patch extraction parameters
	size0 = (64, 64)
	size1 = (128, 128)
	size2 = (256, 256)
	n_labels = 2
	patch_size = size0
	max_size = size2
	

	data_iterator = ITERATOR(data_path, mode = 'train', batch_size = batch_size, num_threads=1)
	data_iterator.mode = 'train'
	# print (data_iterator.train_steps, data_iterator.valid_steps)
	for i in range(10000):
		input_batch, target_batch, wmap_batch = data_iterator.getNextBatch()
		if type(input_batch) is np.ndarray:
			imshow(input_batch[0,:,:,0], target_batch[0,:,:], wmap_batch[0,:,:])
			# if input_batch.shape[0] == 0:
			# 	print input_batch.shape
			# 	import pdb; pdb.set_trace();
		if data_iterator.iter_over == True:
			print ('iterator over')
			data_iterator.reset()
			time.sleep(4)