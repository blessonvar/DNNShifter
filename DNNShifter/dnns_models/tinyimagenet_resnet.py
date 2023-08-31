import torch
import torch.nn as nn
import torch_pruning as tp
from dnns_models import base
import numpy as np
import re

import sys
sys.path.insert(0, '/path/to/this/repo/')

from models import imagenet_resnet, initializers, registry
from datasets import tinyimagenet, registry

class ResNetModel(base.Model):
    def __init__(self, path):
        _weights = self.load_weights(path)
        self.model = self.create_model(_weights)
        super().__init__(self.model)
        self.prune_plan = self.gen_prune_plan()
        
    def load_weights(self, path):
        return torch.load((str(base.base_dir) + str(path) + "model_ep200_it0.pth"),
                          map_location= lambda storage, loc: storage)
    
    def create_model(self, weights):
        model = imagenet_resnet.Model.get_model_from_name('tinyimagenet_resnet_50', initializers.kaiming_normal)
        model.load_state_dict(weights)
        return model
                      
    def gen_prune_plan(self):
        prune_plan = {}
        
        for name, param in self.model.model.named_parameters():
            if re.search('.+weight', name): # or 'block.?.conv?.weight' or 'block.?.shortcut.0.weight' in name:
                cnt = 0
                #layer_idx = int(name.split('.')[1])
                zero_kernels = []
                for group_idx in range(len(param)):
                    group = param[group_idx].cpu().detach().numpy()
                    count = group.size
                    contains_nonzero = np.count_nonzero(group)
                    if contains_nonzero == 0:
                        zero_kernels.append(cnt)
                    cnt = cnt + 1
                prune_plan[name] = zero_kernels
        
        return prune_plan
    
    def prune(self):
        DG = tp.DependencyGraph()
        DG.build_dependency(self.model, example_inputs=torch.randn(1,3,224,224))
        
        pp = self.prune_plan["conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
                      
        pp = self.prune_plan["layer1.0.conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer1[0].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        pp = self.prune_plan["layer1.0.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer1[0].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
#         pp = self.prune_plan["layer1.0.conv3.weight"]
#         pruning_group = DG.get_pruning_group(self.model.model.layer1[0].conv3, tp.prune_conv_out_channels, idxs=pp)
        
#         if DG.check_pruning_group(pruning_group):
#             pruning_group.exec()
            
        pp = self.prune_plan["layer1.1.conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer1[1].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        pp = self.prune_plan["layer1.1.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer1[1].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
#         pp = self.prune_plan["layer1.1.conv3.weight"]
#         pruning_group = DG.get_pruning_group(self.model.model.layer1[1].conv3, tp.prune_conv_out_channels, idxs=pp)
        
#         if DG.check_pruning_group(pruning_group):
#             pruning_group.exec()
            
        pp = self.prune_plan["layer1.2.conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer1[2].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        pp = self.prune_plan["layer1.2.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer1[2].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
#         pp = self.prune_plan["layer1.2.conv3.weight"]
#         pruning_group = DG.get_pruning_group(self.model.model.layer1[2].conv3, tp.prune_conv_out_channels, idxs=pp)
        
#         if DG.check_pruning_group(pruning_group):
#             pruning_group.exec()
            
        pp = self.prune_plan["layer2.0.conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer2[0].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        pp = self.prune_plan["layer2.0.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer2[0].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
#         pp = self.prune_plan["layer2.0.conv3.weight"]
#         pruning_group = DG.get_pruning_group(self.model.model.layer2[0].conv3, tp.prune_conv_out_channels, idxs=pp)
        
#         if DG.check_pruning_group(pruning_group):
#             pruning_group.exec()
            
        pp = self.prune_plan["layer2.1.conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer2[1].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        pp = self.prune_plan["layer2.1.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer2[1].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
#         pp = self.prune_plan["layer2.1.conv3.weight"]
#         pruning_group = DG.get_pruning_group(self.model.model.layer2[1].conv3, tp.prune_conv_out_channels, idxs=pp)
        
#         if DG.check_pruning_group(pruning_group):
#             pruning_group.exec()
            
        pp = self.prune_plan["layer2.2.conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer2[2].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        pp = self.prune_plan["layer2.2.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer2[2].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
#         pp = self.prune_plan["layer2.2.conv3.weight"]
#         pruning_group = DG.get_pruning_group(self.model.model.layer2[2].conv3, tp.prune_conv_out_channels, idxs=pp)
        
#         if DG.check_pruning_group(pruning_group):
#             pruning_group.exec()
            
        pp = self.prune_plan["layer2.3.conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer2[3].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        pp = self.prune_plan["layer2.3.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer2[3].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
#         pp = self.prune_plan["layer2.3.conv3.weight"]
#         pruning_group = DG.get_pruning_group(self.model.model.layer2[3].conv3, tp.prune_conv_out_channels, idxs=pp)
        
#         if DG.check_pruning_group(pruning_group):
#             pruning_group.exec()
            
        pp = self.prune_plan["layer3.0.conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer3[0].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        pp = self.prune_plan["layer3.0.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer3[0].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
#         pp = self.prune_plan["layer3.0.conv3.weight"]
#         pruning_group = DG.get_pruning_group(self.model.model.layer3[0].conv3, tp.prune_conv_out_channels, idxs=pp)
        
#         if DG.check_pruning_group(pruning_group):
#             pruning_group.exec()
            
        pp = self.prune_plan["layer3.1.conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer3[1].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        pp = self.prune_plan["layer3.1.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer3[1].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
#         pp = self.prune_plan["layer3.1.conv3.weight"]
#         pruning_group = DG.get_pruning_group(self.model.model.layer3[1].conv3, tp.prune_conv_out_channels, idxs=pp)
        
#         if DG.check_pruning_group(pruning_group):
#             pruning_group.exec()
            
        pp = self.prune_plan["layer3.2.conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer3[2].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        pp = self.prune_plan["layer3.2.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer3[2].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
#         pp = self.prune_plan["layer3.2.conv3.weight"]
#         pruning_group = DG.get_pruning_group(self.model.model.layer3[2].conv3, tp.prune_conv_out_channels, idxs=pp)
        
#         if DG.check_pruning_group(pruning_group):
#             pruning_group.exec()
            
        pp = self.prune_plan["layer3.3.conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer3[3].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        pp = self.prune_plan["layer3.3.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer3[3].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
#         pp = self.prune_plan["layer3.3.conv3.weight"]
#         pruning_group = DG.get_pruning_group(self.model.model.layer3[3].conv3, tp.prune_conv_out_channels, idxs=pp)
        
#         if DG.check_pruning_group(pruning_group):
#             pruning_group.exec()
            
        pp = self.prune_plan["layer3.4.conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer3[4].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        pp = self.prune_plan["layer3.4.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer3[4].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
#         pp = self.prune_plan["layer3.4.conv3.weight"]
#         pruning_group = DG.get_pruning_group(self.model.model.layer3[4].conv3, tp.prune_conv_out_channels, idxs=pp)
        
#         if DG.check_pruning_group(pruning_group):
#             pruning_group.exec()
            
        pp = self.prune_plan["layer3.5.conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer3[5].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        pp = self.prune_plan["layer3.5.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer3[5].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
#         pp = self.prune_plan["layer3.5.conv3.weight"]
#         pruning_group = DG.get_pruning_group(self.model.model.layer3[5].conv3, tp.prune_conv_out_channels, idxs=pp)
        
#         if DG.check_pruning_group(pruning_group):
#             pruning_group.exec()
            
        pp = self.prune_plan["layer4.0.conv1.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer4[0].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        pp = self.prune_plan["layer4.0.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer4[0].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        #pp = self.prune_plan["layer4.0.conv3.weight"]
        #pruning_group = DG.get_pruning_group(self.model.model.layer4[0].conv3, tp.prune_conv_out_channels, idxs=pp)
        
        #if DG.check_pruning_group(pruning_group):
        #    pruning_group.exec()
            
        #pp = self.prune_plan["layer4.1.conv1.weight"]
        #pruning_group = DG.get_pruning_group(self.model.model.layer4[1].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        #if DG.check_pruning_group(pruning_group):
        #    pruning_group.exec()
            
        pp = self.prune_plan["layer4.1.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer4[1].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        #pp = self.prune_plan["layer4.1.conv3.weight"]
        #pruning_group = DG.get_pruning_group(self.model.model.layer4[1].conv3, tp.prune_conv_out_channels, idxs=pp)
        
        #if DG.check_pruning_group(pruning_group):
        #    pruning_group.exec()
            
        #pp = self.prune_plan["layer4.2.conv1.weight"]
        #pruning_group = DG.get_pruning_group(self.model.model.layer4[2].conv1, tp.prune_conv_out_channels, idxs=pp)
        
        #if DG.check_pruning_group(pruning_group):
        #    pruning_group.exec()
            
        pp = self.prune_plan["layer4.2.conv2.weight"]
        pruning_group = DG.get_pruning_group(self.model.model.layer4[2].conv2, tp.prune_conv_out_channels, idxs=pp)
        
        if DG.check_pruning_group(pruning_group):
            pruning_group.exec()
            
        #pp = self.prune_plan["layer4.2.conv3.weight"]
        #pruning_group = DG.get_pruning_group(self.model.model.layer4[2].conv3, tp.prune_conv_out_channels, idxs=pp)
        
        #if DG.check_pruning_group(pruning_group):
        #    pruning_group.exec()