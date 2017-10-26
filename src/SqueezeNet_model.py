import caffe
from caffe import layers as L, params as P

'''
Function warper
'''
def _conv_relu(bottom, nout, kernel_size, stride, pad, filter_type='xavier'):
    
    if filter_type == 'gaussian':
    	conv = L.Convolution(bottom, num_output=nout, kernel_size=kernel_size, stride=stride, pad=pad, 
    		                 weight_filler=dict(type='gaussian', mean=0, std=0.01))
    else:
    	conv = L.Convolution(bottom, num_output=nout, kernel_size=kernel_size, stride=stride, pad=pad, weight_filler=dict(type='xavier'))
    relu = L.ReLU(conv, in_place=True)

    return conv, relu

'''
Modular blocks for building networks
'''
def _squeeze_block(index, n, bottom, nout_1, nout_2):
	sqz1x1 = 'fire{}/squeeze1x1'.format(index)
	relu_sqz1x1 = 'fire{}/relu_squeeze1x1'.format(index)
	n[sqz1x1], n[relu_sqz1x1] = _conv_relu(bottom, nout_1, 1, 1, 0)

	xpnd1x1 = 'fire{}/expand1x1'.format(index)
	relu_xpnd1x1 = 'fire{}/relu_expand1x1'.format(index)
	n[xpnd1x1], n[relu_xpnd1x1] = _conv_relu(n[relu_sqz1x1], nout_2, 1, 1, 0)

	xpnd3x3 = 'fire{}/expand3x3'.format(index)
	relu_xpnd3x3 = 'fire{}/relu_expand3x3'.format(index)
	n[xpnd3x3], n[relu_xpnd3x3] = _conv_relu(n[relu_sqz1x1], nout_2, 3, 1, 1)

	concat = 'fire{}/concat'.format(index)
	n[concat] = L.Concat(n[xpnd1x1], n[xpnd3x3])

'''
Network Specification
'''
def squeeze_net(split, mean, opt):
    n = caffe.NetSpec()

    # config python data layer
    if split == 'train':
        batch_size = opt.train_batch_size
    if split == 'val':
        batch_size = opt.val_batch_size
    if split == 'test':
        batch_size = opt.test_batch_size

    if split == 'train' or split == 'val':
        dataset_name = opt.train_dataset_name
    else:
        dataset_name = opt.test_dataset_name

    pydata_params = dict(split=split, data_dir=opt.data_dir, 
                         batch_size=batch_size, mean=mean, 
                         dataset=dataset_name, use_HSV=opt.use_HSV, 
                         load_size=opt.load_size, crop_size=opt.crop_size)
    n.data, n.label = L.Python(module='faceData_layers', layer='FaceDataLayer', 
                               ntop=2, param_str=str(pydata_params))

    # start building main body of network
    n.conv1, n.relu_conv1 = _conv_relu(n.data, 64, kernel_size=3, stride=2, pad=0)
    n.pool1 = L.Pooling(n.relu_conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    _squeeze_block('2', n, n.pool1, 16, 64)
    _squeeze_block('3', n, n['fire2/concat'], 16, 64)

    n.pool3 = L.Pooling(n['fire3/concat'], kernel_size=3, stride=2, pool=P.Pooling.MAX)
    
    _squeeze_block('4', n, n.pool3, 32, 128)
    _squeeze_block('5', n, n['fire4/concat'], 32, 128)

    n.pool5 = L.Pooling(n['fire5/concat'], kernel_size=3, stride=2, pool=P.Pooling.MAX)

    _squeeze_block('6', n, n.pool5, 48, 192)
    _squeeze_block('7', n, n['fire6/concat'], 48, 192)
    _squeeze_block('8', n, n['fire7/concat'], 64, 256)
    _squeeze_block('9', n, n['fire8/concat'], 64, 256)

    n.drop9 = L.Dropout(n['fire9/concat'], dropout_ratio=0.5, in_place=True)

    n.conv10, n.relu_conv10 = _conv_relu(n.drop9, 1000, kernel_size=1, stride=1, pad=0)
    n.pool10 = L.Pooling(n.relu_conv10, global_pooling=True, pool=P.Pooling.AVE)   

    # fully connected classifier
    lr_ratio = 10 # lr multiplier for truncated layers
    n.fc_face = L.InnerProduct(n.pool10, num_output=2, 
                                param=[dict(lr_mult=1*lr_ratio, decay_mult=1), dict(lr_mult=2*lr_ratio, decay_mult=0)], 
                                weight_filler=dict(type='gaussian', std=0.01), 
                                bias_filler=dict(type='constant', value=0)
                                )

    # loss and accuracy layer
    n.loss = L.SoftmaxWithLoss(n.fc_face, n.label)
    n.acc = L.Accuracy(n.fc_face, n.label)
    return n.to_proto()
