import torch
from dnns_models import base
import numpy as np

import sys
sys.path.insert(0, '/path/to/this/repo/')

from models import cifar_vgg, initializers, registry
from datasets import cifar10, registry

class VGGModel(base.Model):
    def __init__(self, path, fuse_bn=True):
        _weights = self.load_weights(path)
        self.model = self.create_model(_weights)
        if fuse_bn:
            self.fuse_conv_bn()
        super().__init__(self.model)
        self.prune_plan = self.gen_prune_plan()
    
    def load_weights(self, path):
        return torch.load((str(base.base_dir) + str(path) + "model_ep160_it0.pth"),
                          map_location= lambda storage, loc: storage)
    
    def create_model(self, weights):
        model = cifar_vgg.Model.get_model_from_name('cifar_vgg_16', initializers.kaiming_normal)
        model.load_state_dict(weights)
        return model
    
    def fuse_conv_bn(self): #Concrete implementation
        features = list(self.model._modules.items())
        seq_layers = features[0][1]
    
        for layer_idx in range(len(seq_layers)):
            if not isinstance(seq_layers[layer_idx], torch.nn.modules.pooling.MaxPool2d):
                seq_layers[layer_idx].conv = super().fuse_conv_and_bn(
                    seq_layers[layer_idx].conv,
                    seq_layers[layer_idx].bn)
                del seq_layers[layer_idx].bn
                
    def prune(self):
        prune_plan = self.prune_plan

        layer_idexes = list(prune_plan.keys())

        for i in range(len(layer_idexes)):
            curr_layer_idx = layer_idexes[i]
            plan = {}
            if curr_layer_idx == 0:
                plan = {curr_layer_idx: prune_plan[curr_layer_idx]}
            else:
                prev_layer_idx = layer_idexes[(i-1)]
                plan = {prev_layer_idx: prune_plan[prev_layer_idx], curr_layer_idx: prune_plan[curr_layer_idx]}
            
        
            last_layer = False
            if (i+1) == len(layer_idexes):
                last_layer = True
    
            self.modify_conv(plan, last_layer)
    
    def gen_prune_plan(self):
        prune_plan = {}
        
        for name, param in self.model.named_parameters():
            if 'conv.weight' in name:
                cnt = 0
                layer_idx = int(name.split('.')[1])
                zero_kernels = []
                for group_idx in range(len(param)):
                    group = param[group_idx].cpu().detach().numpy()
                    count = group.size
                    contains_nonzero = np.count_nonzero(group)
                    if contains_nonzero == 0:
                        zero_kernels.append(cnt)
                    cnt = cnt + 1
                prune_plan[layer_idx] = zero_kernels
        
        return prune_plan