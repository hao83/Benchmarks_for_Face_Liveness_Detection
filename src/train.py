import os
import sys
import caffe
import net
import solver

import numpy as np
import argparse
import time
import pdb

from util import get_mean_all, get_mean_dataset

'''
Define options
'''
parser = argparse.ArgumentParser()

# Training Policy
parser.add_argument('--base_lr',   type=float, default=1e-8,    help='base learning rate')
parser.add_argument('--lr_policy', type=str,   default='fixed', help='learning rate manipulation policy')
parser.add_argument('--stepsize',  type=int,   default=5000,    help='lr drop stepsize')
parser.add_argument('--gamma',     type=float, default=0.1,     help='ratio for lr drop at each step')
parser.add_argument('--train_batch_size', type=int, default=20,  help='batch size for training')
parser.add_argument('--val_batch_size',   type=int, default=20, help='batch size for validation')
parser.add_argument('--num_epoch',     type=int, default=10, help='number of epochs')
parser.add_argument('--test_interval', type=int, default=200, help='interval for invoking validation/testing')

# Model and Dataset
parser.add_argument('--load_size',  type=int, default=256, help='default load size of training samples')
parser.add_argument('--crop_size',  type=int, default=224, help='default crop size of training samples')
parser.add_argument('--data_dir',   type=str, default='../data', help='path to the data folder')
parser.add_argument('--model_name', type=str, default='VggFace', help='name of model to be used for training')
parser.add_argument('--train_dataset_name', type=str, default='MZDX', help='name of dataset to be used for training')

# Ohers
parser.add_argument('--exp_suffix', type=str, default='', help='prefix to distinguish between different experiment settings')
parser.add_argument('--use_HSV', action='store_true', help='indicator of using HSV colorspace (set to use)')
parser.add_argument('--gpu_id',  type=int, default=0, help='the id of gpu')

opt = parser.parse_args()

'''
Initialization Setup
'''
caffe.set_device(opt.gpu_id)
caffe.set_mode_gpu()

# set paths
model_path = os.path.join('..', 'model', opt.model_name)
HSV_suffix = '_HSV' if opt.use_HSV else ''
if not opt.exp_suffix == '':
	opt.exp_suffix = '_' + opt.exp_suffix
suffix = HSV_suffix + opt.exp_suffix # suffix to distinguish models over experiments

if opt.train_dataset_name == 'all':
	train_prototxt_path = "{}/train{}.prototxt".format(model_path, HSV_suffix)
	val_prototxt_path   = "{}/val{}.prototxt".format(model_path, HSV_suffix)
	solver_path         = "{}/solver{}.prototxt".format(model_path, HSV_suffix)
	snapshot_path       = "{}/snapshots/all{}".format(model_path, suffix)	
else:
	model_dataset_path = os.path.join(model_path, opt.train_dataset_name)
	model_dataset_snapshots_path = os.path.join(model_dataset_path, 'snapshots')
	
	train_prototxt_path = "{}/train{}.prototxt".format(model_dataset_path, HSV_suffix)
	val_prototxt_path   = "{}/val{}.prototxt".format(model_dataset_path, HSV_suffix)
	solver_path         = "{}/solver{}.prototxt".format(model_dataset_path, HSV_suffix)
	snapshot_path       = "{}/{}".format(model_dataset_snapshots_path, suffix)

# Compute mean value
# all:      dataset_mean = (86.675902, 100.892992, 133.855434)   dataset_size = 342559
# MZDX:     dataset_mean = (92.308940, 108.304358, 141.335537)   dataset_size = 164290
# NUAA:     dataset_mean = (111.607392, 115.243609, 125.662128)  dataset_size = 8411
# MSU-MFSD: dataset_mean = (101.325867, 115.503758, 141.250656)  dataset_size = 25750
# cbsr:     dataset_mean = (93.745483, 100.534902, 123.578854)   dataset_size = 36851
# replayattack: dataset_mean = (70.146368, 85.030646, 124.795727), dataset_size = 107257

# MZDX_eye:  dataset_mean = (81.237413, 95.646728, 131.313932)  dataset_size = 164976
# MZDX(HSV): dataset_mean = (27.426452, 97.793206, 142.434513)  dataset_size = 164290

dataset_mean = (86.675902, 100.892992, 133.855434)
dataset_size = 342559

'''
if opt.train_dataset_name == 'all':
	mean, dataset_size = get_mean_all(opt)
else:
	mean, dataset_size = get_mean_dataset(opt, opt.train_dataset_name)
dataset_mean = (mean[0], mean[1], mean[2]) # returned 'mean' is an array, need to be converted to tuple
'''

'''
Make Prototxt files
'''
# make prototxt
net.make_net(train_prototxt_path, 'train', dataset_mean, opt)
net.make_net(val_prototxt_path,   'val',   dataset_mean, opt)
solver.make_solver(train_prototxt_path, val_prototxt_path, 
				   solver_path, snapshot_path, opt, dataset_size)

# Read in solver and pre-trained parameters
mySolver = caffe.get_solver(solver_path)
caffemodel_path = "{}/{}_model/{}.caffemodel".format(model_path, opt.model_name, opt.model_name)
mySolver.net.copy_from(caffemodel_path)

'''
Start Training
'''
# Ordinary train loop
train_loss = np.zeros(mySolver.param.max_iter)
val_acc    = np.zeros(int(np.ceil(mySolver.param.max_iter/opt.test_interval)) + 1)
print 'Total number of iterations: {}'.format(mySolver.param.max_iter)
for iter in range(mySolver.param.max_iter):
	mySolver.step(1)

	# store the training loss
	train_loss[iter] = mySolver.net.blobs['loss'].data

	# store the validation accuracy
	if iter % opt.test_interval == 0:
		acc = mySolver.test_nets[0].blobs['acc'].data
		val_acc[iter // opt.test_interval] = acc

# log the train error and val acc
localtime  = time.localtime()
timeString = time.strftime("%Y_%m_%d_%H_%M_%S", localtime)

train_error_save_path = '../results/%s_train_error_' % opt.train_dataset_name + timeString + '.txt'
with open (train_error_save_path, 'w') as f:
	for loss in train_loss:
		f.write('%s\n' % str(loss)) 

val_acc_save_path =  '../results/%s_val_acc_' % opt.train_dataset_name + timeString + '.txt'
with open (val_acc_save_path, 'w') as f:
	for acc in val_acc:
		f.write('%s\n' % str(acc)) 