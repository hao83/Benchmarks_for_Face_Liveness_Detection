import os
import sys
import caffe
import net
import solver

import numpy as np
import argparse
import time
import pdb

# helper function for reading mean value and size for training set
def GetDatasetMean(opt):
	mean_file_path = os.path.join(opt.data_dir, opt.train_dataset_name, 'trainset_mean.txt')
	f = open(mean_file_path)
	line = f.readline()
	f.close()
	nums = line.split(' ')
	dataset_size = int(nums[0])
	dataset_mean = (float(nums[1]), float(nums[2]), float(nums[3]))

	return dataset_size, dataset_mean

'''
Define options
'''
parser = argparse.ArgumentParser()

# Training Policy
parser.add_argument('--base_lr',   type=float, default=1e-8,    help='base learning rate')
parser.add_argument('--lr_policy', type=str,   default='step', help='learning rate manipulation policy')
parser.add_argument('--stepsize',  type=int,   default=5000,    help='lr drop stepsize')
parser.add_argument('--gamma',     type=float, default=0.1,     help='ratio for lr drop at each step')
parser.add_argument('--train_batch_size', type=int, default=10,  help='batch size for training')
parser.add_argument('--num_epoch',     type=int, default=10, help='number of epochs')

# Model and Dataset
parser.add_argument('--load_size',  type=int, default=256, help='default load size of training samples')
parser.add_argument('--crop_size',  type=int, default=224, help='default crop size of training samples')
parser.add_argument('--data_dir',   type=str, default='../data', help='path to the data folder')
parser.add_argument('--model_name', type=str, default='VggFace', help='name of model to be used for training')
parser.add_argument('--train_dataset_name', type=str, default='replayattack', help='name of dataset to be used for training')

# Ohers
parser.add_argument('--exp_suffix', type=str, default='', help='prefix to distinguish between different experiment settings')
parser.add_argument('--gpu_id',  type=int, default=0, help='the id of gpu')

opt = parser.parse_args()

'''
Initialization Setup
'''
caffe.set_device(opt.gpu_id)
caffe.set_mode_gpu()

# set paths
model_path = os.path.join('..', 'model', opt.model_name)
if not opt.exp_suffix == '':
	opt.exp_suffix = '_' + opt.exp_suffix

model_dataset_path = os.path.join(model_path, opt.train_dataset_name)
if not os.path.exists(model_dataset_path):
	os.mkdir(model_dataset_path)

caffemodel_path = "{}/{}_model/{}.caffemodel".format(model_path, opt.model_name, opt.model_name)
train_prototxt_path = "{}/train.prototxt".format(model_dataset_path)
solver_path         = "{}/solver.prototxt".format(model_dataset_path)
model_dataset_snapshots_path = os.path.join(model_dataset_path, 'snapshots')
if not os.path.exists(model_dataset_snapshots_path):
	os.mkdir(model_dataset_snapshots_path)

# get the mean value of training set
dataset_size, dataset_mean = GetDatasetMean(opt)

'''
Make Prototxt files
'''
# make prototxt
net.make_net(train_prototxt_path, 'train', dataset_mean, opt)
solver.make_solver(train_prototxt_path, solver_path, model_dataset_snapshots_path, opt, dataset_size)

# Read in solver and pre-trained parameters
mySolver = caffe.get_solver(solver_path)
mySolver.net.copy_from(caffemodel_path)

'''
Start Training
'''
# Ordinary train loop
train_loss = np.zeros(mySolver.param.max_iter)
train_acc  = np.zeros(mySolver.param.max_iter)
print 'Total number of iterations: {}'.format(mySolver.param.max_iter)
for iter in range(mySolver.param.max_iter):
	mySolver.step(1)

	# store the training loss
	train_loss[iter] = mySolver.net.blobs['loss'].data
	train_acc[iter]  = mySolver.net.blobs['acc'].data

	pdb.set_trace()

# log the train error and val acc
localtime  = time.localtime()
timeString = time.strftime("%Y_%m_%d_%H_%M_%S", localtime)

train_error_save_path = '../results/%s_train_error_' % opt.train_dataset_name + timeString + '.txt'
with open (train_error_save_path, 'w') as f:
	for loss in train_loss:
		f.write('loss = %s, acc = \n' % str(loss)) 