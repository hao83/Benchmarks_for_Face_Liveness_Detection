from VggFace_model import vgg_face
from ResNet_model  import resnet50, resnet18
from SqueezeNet_model import squeeze_net

def make_net(net_path, split, mean, opt):
    '''
    net_path: path for the prototxt file of the net
    split: 'train' / 'test'
    mean: channel mean for the datasets used   
    '''
    print "Writing prototxt file for train net..."
    with open(net_path, 'w') as f:
        if opt.model_name == 'VggFace':
            f.write( str ( vgg_face(split, mean, opt) ) )
        elif opt.model_name == 'ResNet50':
            f.write( str ( resnet50(split, mean, opt) ) )
        elif opt.model_name == 'ResNet18':
            f.write( str ( resnet18(split, mean, opt) ) )
        elif opt.model_name == 'SqueezeNet':
            f.write( str ( squeeze_net(split, mean, opt) ) )
        else:
            raise ValueError('Unrecognized network type')