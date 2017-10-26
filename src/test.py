import sys
import caffe
import numpy as np
import net
import solver
from PIL import Image

import pdb

import argparse
from util import get_size_dataset, get_size_all, get_mean_dataset

'''
Define options
'''
parser = argparse.ArgumentParser()

parser.add_argument('--load_size',  type=int, default=256, help='default load size of training samples')
parser.add_argument('--crop_size',  type=int, default=224, help='default crop size of training samples')
parser.add_argument('--test_batch_size',    type=int, default=100, help='batch size for testing')

parser.add_argument('--train_dataset_name', type=str, default='MZDX',  help='name of dataset used for training')
parser.add_argument('--test_dataset_name',  type=str, default='MZDX',  help='name of dataset to be used for testing')
parser.add_argument('--data_dir',     type=str, default='../data', help='path to the ROOT data folder (disregard the model and dataset used)')
parser.add_argument('--model_name',   type=str, default='VggFace', help='name of model to be used for training')
parser.add_argument('--model_iter',   type=int, default=40000, help='iteration number for the model loaded')

parser.add_argument('--exp_suffix', type=str, default='', help='prefix to distinguish between different experiment settings')
parser.add_argument('--use_HSV',  action='store_true', help='indicator of using HSV colorspace (set to use)')
parser.add_argument('--gpu_id',  type=int, default=0,  help='the id of gpu')

opt = parser.parse_args()

'''
Initialization Setup
'''
caffe.set_device(opt.gpu_id)
caffe.set_mode_gpu()

# define path
HSV_suffix = '_HSV' if opt.use_HSV else ''
if not opt.exp_suffix == '':
	opt.exp_suffix = '_' + opt.exp_suffix
suffix = HSV_suffix + opt.exp_suffix # suffix to distinguish models over experiments

if opt.train_dataset_name == 'all':
	snapshot_path = '../model/{}/snapshots'.format(opt.model_name)
	test_prototxt_path = '../model/{}/test{}.prototxt'.format(opt.model_name, suffix)
else:
	snapshot_path = '../model/{}/{}/snapshots'.format(opt.model_name, opt.train_dataset_name)
	test_prototxt_path = '../model/{}/{}/test{}.prototxt'.format(opt.model_name, opt.train_dataset_name, suffix)	

'''
Make Prototxt files
'''
# make net
# all:      dataset_mean = (86.675902, 100.892992, 133.855434)   dataset_size = 342559
# MZDX:     dataset_mean = (92.308940, 108.304358, 141.335537)   dataset_size = 164290
# NUAA:     dataset_mean = (111.607392, 115.243609, 125.662128)  dataset_size = 8411
# MSU-MFSD: dataset_mean = (101.325867, 115.503758, 141.250656)  dataset_size = 25750
# cbsr:     dataset_mean = (93.745483, 100.534902, 123.578854)   dataset_size = 36851
# replayattack: dataset_mean = (70.146368, 85.030646, 124.795727), dataset_size = 107257

# MZDX_eye:  dataset_mean = (81.237413, 95.646728, 131.313932)  dataset_size = 164976
train_dataset_mean = (86.675902, 100.892992, 133.855434)
net.make_net(test_prototxt_path, 'test', train_dataset_mean, opt)

# load net
model_name = '{}/{}_iter_{}.caffemodel'.format(snapshot_path, opt.train_dataset_name + HSV_suffix, opt.model_iter)
net = caffe.Net(test_prototxt_path, model_name, caffe.TEST)

'''
Start Testing
'''
if opt.test_dataset_name == 'all':
	num_test_samples = get_size_all(opt)
else:
	num_test_samples = get_size_dataset(opt, opt.train_dataset_name)
niter = int( np.floor( 1.0*num_test_samples/opt.test_batch_size ) )

correct_cnt = 0
total_cnt = 0
labels_all = []
preds_all  = []
for iter in range(niter):

	net.forward()

	# switch final layer name
	if opt.model_name == 'VggFace':
		pred_layer_name = 'fc9_face'
	elif opt.model_name == 'ResNet50' or opt.model_name == 'ResNet18':
		pred_layer_name = 'fc_face2'
	elif opt.model_name == 'SqueezeNet':
		pred_layer_name = 'fc_face'


	preds  = net.blobs[pred_layer_name].data.argmax(axis=1)
	labels = np.ndarray.flatten(net.blobs['label'].data)

	preds_all  += preds.tolist()
	labels_all += labels.tolist()

	correct_cnt += np.sum(preds == np.ndarray.flatten(labels))
	total_cnt += opt.test_batch_size
	acc = 1.0 * correct_cnt / total_cnt

	print 'Progress: {}/{}; Accuracy: {} ({}/{})'.format(iter+1, niter, 
													acc, correct_cnt, total_cnt)

total_acc = 1.0 * correct_cnt / total_cnt
print 'Total accuracy is {} ({}/{})'.format(total_acc, correct_cnt, total_cnt)

# compute FPR and FNR
TN = 0
TP = 0
FN = 0
FP = 0
for i in xrange(len(labels_all)):
	if labels_all[i] == 1 and preds_all[i] == 1:
		TP += 1
	if labels_all[i] == 1 and preds_all[i] == 0:
		FN += 1
	if labels_all[i] == 0 and preds_all[i] == 1:
		FP += 1
	if labels_all[i] == 0 and preds_all[i] == 0:
		TN +=1

FPR = 1.0 * FP / (TN + FP)
FNR = 1.0 * FN / (TP + FN)
print 'False Positive Rate (FPR) is {} ({}/{})'.format(FPR, FP, (TN+FP))
print 'False Negative Rate (FNR) is {} ({}/{})'.format(FNR, FN, (TP+FN))
