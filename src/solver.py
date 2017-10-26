from caffe.proto import caffe_pb2
import math
import pdb

def make_solver(train_net_path, val_net_path, solver_path, snapshot_path, opt, dataset_size):
	s = caffe_pb2.SolverParameter()

	# specify locations of the train and test networks.
	s.train_net = train_net_path
	s.test_net.append(val_net_path)

	# specify parameters for iterations
	s.test_interval = opt.test_interval # interval for invoking testing
	s.test_iter.append(opt.val_batch_size) # number of batches used for testing

	s.max_iter = int(opt.num_epoch * dataset_size / opt.train_batch_size) 

	# specify parameters for learning policy
	s.base_lr = opt.base_lr
	s.lr_policy = opt.lr_policy
	if s.lr_policy == 'step':
		s.gamma = opt.gamma
		s.stepsize = opt.stepsize

	s.type = "Adam"
	s.momentum = 0.9
	s.weight_decay= 5e-4
	s.iter_size = 1 # no gradient accumulation

	# specify other helper parameters
	s.display = 20
	s.snapshot = 2500
	s.snapshot_prefix = snapshot_path
	s.solver_mode = caffe_pb2.SolverParameter.GPU

	print "Writing prototxt file for solver..."
	with open(solver_path, 'w') as f:
		f.write(str(s))