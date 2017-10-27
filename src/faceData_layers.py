import caffe

import os
import numpy as np
from PIL import Image
import copy
import random
import cv2

import pdb

"""
Reference: 
https://github.com/BVLC/caffe/blob/master/examples/pycaffe/layers/pascal_multilabel_datalayers.py
https://github.com/yunfan0621/fcn.berkeleyvision.org/blob/master/voc_layers.py
"""

class FaceDataLayer(caffe.Layer):
	"""
	Load (face image, face label) pairs from created dataset
	The input image size is fixed to be 3 x 224 x 224 with 2 labels
	"""

	def setup(self, bottom, top):

		self.top_names = ['data', 'label']

		# start config
		params = eval(self.param_str) # do param_str in python console (create dict)

		self.batch_size = params['batch_size']
		self.batch_loader = BatchLoader(params) # save all other params to loader 

		"""
		since we use a fixed input image size, reshape the data layer only once
		and save it from be called every time in reshape()
		"""
		top[0].reshape(self.batch_size, 3, self.batch_loader.crop_size, self.batch_loader.crop_size)
		top[1].reshape(self.batch_size, 1)

	def forward(self, bottom, top):
		# assign output for top blob
		for iter in range(self.batch_size):
			# load (image, label) pair via batch loader
			# NOTE: only one single pair is loaded at a time (not a batch)
			im, label = self.batch_loader.load_next_pair()

			# assign data and label to data layer
			top[0].data[iter, ...] = im
			top[1].data[iter, ...] = label

	def backward(self, top, propagate_down, bottom):
		pass

	def reshape(self, bottom, top):
		# reshaping done in layer setup
		pass


class BatchLoader(object):
	
	"""
    This class abstracts away the loading of images (for the ease of debugging)
    Images are loaded individually (one at a time).
    """

	def __init__(self, params):
		# load image paths and their labels (both are list of strings)

		self.data_dir   = params['data_dir']
		self.dataset    = params['dataset']
		self.split      = params['split']
		self.batch_size = params['batch_size']
		self.load_size  = params['load_size']
		self.crop_size  = params['crop_size']
		self.mean = np.array(params['mean'])

		# using a specific dataset
		self.image_paths = open('{}/{}/{}_image_list.txt'.format(self.data_dir, self.dataset, self.split)).read().splitlines()
		self.labels      = open('{}/{}/{}_label_list.txt'.format(self.data_dir, self.dataset, self.split)).read().splitlines()			

		print 'Shuffling dataset before training...'
		self._cur = 0 # index of current image
		self.shuffle_dataset()

	def load_next_pair(self):
		# return the next batch of (image, label) pairs

		# check whether an epoch has been finished
		if self._cur >= len(self.image_paths):
			self._cur = 0
			self.shuffle_dataset()

		#  im_PIL = Image.open(self.image_paths[self._cur])

		im = cv2.imread(os.path.join(self.data_dir, self.dataset, self.image_paths[self._cur])) # we switch to use OpenCV to load images

		# data augmentation
		im = self.data_augment(im)
		label = int(self.labels[self._cur])

		self._cur += 1
		return self.preprocessor(im), label

	def shuffle_dataset(self):
		# shuffle the dataset

		# shuffle the image path and label using the same permutation
		permute_idx = np.random.permutation(len(self.image_paths))
		image_paths_tmp = [self.image_paths[i] for i in permute_idx]
		labels_tmp = [self.labels[i] for i in permute_idx]
		
		self.image_paths = copy.copy(image_paths_tmp)
		self.labels = copy.copy(labels_tmp)

		# reset the offset (offset reset outside this functionality)
		# self._cur = 0

	def data_augment(self, im):
		(nrow, ncol, nchannel) = im.shape

		if self.split == 'train':
			# do data augmentation in training phase
			if nrow != self.load_size or ncol != self.load_size:
				im = cv2.resize(im, (self.load_size, self.load_size))

			
			# # convert the image to array and move on
			# im = np.asarray(im_PIL)
			
			# do a random crop as data augmentation
			max_offset = self.load_size - self.crop_size # 256 -224
			w_offset = random.randint(0, max_offset)
			h_offset = random.randint(0, max_offset)		
			im = im[h_offset:h_offset+self.crop_size, w_offset:w_offset+self.crop_size,:]

			# do a simple horizontal flip as data augmentation
			flip = np.random.choice(2)*2-1
			im = im[:, ::flip, :]
		else:
			# simply forward the image for testing or validation
			if nrow != self.crop_size or ncol != self.crop_size:
				im = cv2.resize(im, (self.crop_size, self.crop_size))

			# im = np.asarray(im_PIL)

		return im			

	def preprocessor(self, im):
		"""
		preprocess the image for caffe use:
		- cast to float
		# we do not do this anymore whenever using OpenCV ==> - switch channels RGB -> BGR 
		- subtract mean
		- transpose to channel x height x width order
		"""

		im = np.array(im, dtype=np.float32)
		# im = im[:,:,::-1] # RGB -> BGR
		im -= self.mean # the order of mean should be BGR
		im = im.transpose((2, 0, 1))
		return im