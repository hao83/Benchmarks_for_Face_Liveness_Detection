import caffe
from caffe import layers as L, params as P

'''
Function warper
'''
def _conv_bn_scale(bottom, nout, bias_term=False, **kwargs):
    # build a conv -> BN -> relu block
    
    conv  = L.Convolution(bottom, num_output=nout, bias_term=bias_term, **kwargs)
    bn    = L.BatchNorm(conv, use_global_stats=True, in_place=True)
    scale = L.Scale(bn, bias_term=True, in_place=True)

    return conv, bn, scale

'''
Modular blocks for building networks
'''
def _resnet_block_2stages(name, n, bottom, nout, branch1=False, initial_stride=2):
    # Basic ResNet block for ResNet-18
    
    if branch1:
        res_b1   = 'res{}_branch1'.format(name)
        bn_b1    = 'bn{}_branch1'.format(name)
        scale_b1 = 'scale{}_branch1'.format(name)
        n[res_b1], n[bn_b1], n[scale_b1] = _conv_bn_scale(
                   bottom, nout, kernel_size=1, stride=initial_stride, pad=0)
    else:
        initial_stride = 1

    res   = 'res{}_branch2a'.format(name)
    bn    = 'bn{}_branch2a'.format(name)
    scale = 'scale{}_branch2a'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(bottom, nout, kernel_size=3, stride=initial_stride, pad=1)
    relu2a    = 'res{}_branch2a_relu'.format(name)
    n[relu2a] = L.ReLU(n[scale], in_place=True)
    
    res   = 'res{}_branch2b'.format(name)
    bn    = 'bn{}_branch2b'.format(name)
    scale = 'scale{}_branch2b'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(n[relu2a], nout, kernel_size=3, stride=1, pad=1)
    res   = 'res{}'.format(name)


    # add up the 'identity' branch
    if branch1:
        n[res] = L.Eltwise(n[scale_b1], n[scale])
    else:
        n[res] = L.Eltwise(bottom, n[scale])
    
    relu    = 'res{}_relu'.format(name)
    n[relu] = L.ReLU(n[res], in_place=True)

def _resnet_block_3stages(name, n, bottom, nout, branch1=False, initial_stride=2):
    # Basic ResNet block for ResNet-50
    
    if branch1:
        res_b1   = 'res{}_branch1'.format(name)
        bn_b1    = 'bn{}_branch1'.format(name)
        scale_b1 = 'scale{}_branch1'.format(name)
        n[res_b1], n[bn_b1], n[scale_b1] = _conv_bn_scale(
                   bottom, 4*nout, kernel_size=1, stride=initial_stride, pad=0)
    else:
        initial_stride = 1

    res   = 'res{}_branch2a'.format(name)
    bn    = 'bn{}_branch2a'.format(name)
    scale = 'scale{}_branch2a'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(bottom, nout, kernel_size=1, stride=initial_stride, pad=0)
    relu2a    = 'res{}_branch2a_relu'.format(name)
    n[relu2a] = L.ReLU(n[scale], in_place=True)
    
    res       = 'res{}_branch2b'.format(name)
    bn        = 'bn{}_branch2b'.format(name)
    scale     = 'scale{}_branch2b'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(n[relu2a], nout, kernel_size=3, stride=1, pad=1)    
    relu2b    = 'res{}_branch2b_relu'.format(name)
    n[relu2b] = L.ReLU(n[scale], in_place=True)
    
    res   = 'res{}_branch2c'.format(name)
    bn    = 'bn{}_branch2c'.format(name)
    scale = 'scale{}_branch2c'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(n[relu2b], 4*nout, kernel_size=1, stride=1, pad=0)
    res   = 'res{}'.format(name)

    # add up the 'identity' branch (L.Eltwise use SUM operation by default)
    if branch1:
        n[res] = L.Eltwise(n[scale_b1], n[scale])
    else:
        n[res] = L.Eltwise(bottom, n[scale])
    
    relu    = 'res{}_relu'.format(name)
    n[relu] = L.ReLU(n[res], in_place=True)


'''
Network Specifiction
'''
def resnet18(split, mean, opt):
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
    # There main differences:
    #   1. do not use 4*nout for certain convolution layers
    #   2. do not use bias_term for convolution layer before start of residual blocks
    #   3. do not set the BN layer parameter, moving_average_fraction, to 0.9 (using default value 0.999)
    #   4. for weight filter initialziation, we do not specify 'msra'
    n.conv1, n.bn_conv1, n.scale_conv1 = _conv_bn_scale(n.data, 64, bias_term=False, kernel_size=7, pad=3, stride=2)
    n.conv1_relu = L.ReLU(n.scale_conv1, in_place=True)
    n.pool1 = L.Pooling(n.conv1_relu, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    _resnet_block_2stages('2a', n, n.pool1, 64, branch1=True, initial_stride=1)
    _resnet_block_2stages('2b', n, n.res2a_relu, 64)

    _resnet_block_2stages('3a', n, n.res2b_relu, 128, branch1=True)
    _resnet_block_2stages('3b', n, n.res3a_relu, 128)

    _resnet_block_2stages('4a', n, n.res3b_relu, 256, branch1=True)
    _resnet_block_2stages('4b', n, n.res4a_relu, 256)

    _resnet_block_2stages('5a', n, n.res4b_relu, 512, branch1=True)
    _resnet_block_2stages('5b', n, n.res5a_relu, 512)

    n.pool5 = L.Pooling(n.res5b_relu, kernel_size=7, stride=1, pool=P.Pooling.AVE)

    # fully connected classifier
    lr_ratio = 100 # lr multiplier for truncated layers
    n.fc_face1 = L.InnerProduct(n.pool5, num_output=1000, 
                                param=[dict(lr_mult=1*lr_ratio, decay_mult=1), dict(lr_mult=2*lr_ratio, decay_mult=0)], 
                                weight_filler=dict(type='gaussian', std=0.01), 
                                bias_filler=dict(type='constant', value=0)
                                )
    n.fc_face2 = L.InnerProduct(n.fc_face1, num_output=2, 
                                param=[dict(lr_mult=1*lr_ratio, decay_mult=1), dict(lr_mult=2*lr_ratio, decay_mult=0)], 
                                weight_filler=dict(type='gaussian', std=0.01), 
                                bias_filler=dict(type='constant', value=0)
                                )   

    # loss and accuracy layer
    n.loss = L.SoftmaxWithLoss(n.fc_face2, n.label)
    n.acc = L.Accuracy(n.fc_face2, n.label)
    return n.to_proto()

def resnet50(split, mean, opt):
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
    n.conv1, n.bn_conv1, n.scale_conv1 = _conv_bn_scale(n.data, 64, bias_term=True, kernel_size=7, pad=3, stride=2)
    n.conv1_relu = L.ReLU(n.scale_conv1)
    n.pool1 = L.Pooling(n.conv1_relu, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    _resnet_block_3stages('2a', n, n.pool1, 64, branch1=True, initial_stride=1)
    _resnet_block_3stages('2b', n, n.res2a_relu, 64)
    _resnet_block_3stages('2c', n, n.res2b_relu, 64)

    _resnet_block_3stages('3a', n, n.res2c_relu, 128, branch1=True)
    _resnet_block_3stages('3b', n, n.res3a_relu, 128)
    _resnet_block_3stages('3c', n, n.res3b_relu, 128)
    _resnet_block_3stages('3d', n, n.res3c_relu, 128)

    _resnet_block_3stages('4a', n, n.res3d_relu, 256, branch1=True)
    _resnet_block_3stages('4b', n, n.res4a_relu, 256)
    _resnet_block_3stages('4c', n, n.res4b_relu, 256)
    _resnet_block_3stages('4d', n, n.res4c_relu, 256)
    _resnet_block_3stages('4e', n, n.res4d_relu, 256)
    _resnet_block_3stages('4f', n, n.res4e_relu, 256)

    _resnet_block_3stages('5a', n, n.res4f_relu, 512, branch1=True)
    _resnet_block_3stages('5b', n, n.res5a_relu, 512)
    _resnet_block_3stages('5c', n, n.res5b_relu, 512)

    n.pool5 = L.Pooling(n.res5c_relu, kernel_size=7, stride=1, pool=P.Pooling.AVE)

    # fully connected classifier
    lr_ratio = 100 # lr multiplier for truncated layers
    n.fc_face1 = L.InnerProduct(n.pool5, num_output=1000, 
                                param=[dict(lr_mult=1*lr_ratio, decay_mult=1), dict(lr_mult=2*lr_ratio, decay_mult=0)], 
                                weight_filler=dict(type='gaussian', std=0.01), 
                                bias_filler=dict(type='constant', value=0)
                                )
    n.fc_face2 = L.InnerProduct(n.fc_face1, num_output=2, 
                                param=[dict(lr_mult=1*lr_ratio, decay_mult=1), dict(lr_mult=2*lr_ratio, decay_mult=0)], 
                                weight_filler=dict(type='gaussian', std=0.01), 
                                bias_filler=dict(type='constant', value=0)
                                )   

    # loss and accuracy layer
    n.loss = L.SoftmaxWithLoss(n.fc_face2, n.label)
    n.acc = L.Accuracy(n.fc_face2, n.label)
    return n.to_proto()