import torch
import numpy as np

base_dir = "/path/to/sparse/models/"
class Model:
    def __init__(self, model):
        self.model = model
        del self.model.criterion
        
    def fuse_conv_and_bn(self, conv, bn):
        fusedconv = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True
        )
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
        with torch.no_grad():
            fusedconv.weight.copy_( torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )
            if conv.bias is not None:
                b_conv = conv.bias
            else:
                b_conv = torch.zeros( conv.weight.size(0) )
            b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
            fusedconv.bias.copy_( torch.matmul(w_bn, b_conv) + b_bn )
        return fusedconv
    
    def modify_conv(self, conv_filters, last_layer=False):
        layer_indexes = list(conv_filters.keys())
        prev_layer = -1
        prev_layer_filters = []
        curr_layer = layer_indexes[-1]
        curr_layer_filters = conv_filters[curr_layer]
        
        conv = self.model.layers[curr_layer].conv

        if len(layer_indexes) > 1:
            prev_layer = layer_indexes[-2]
            prev_layer_filters = conv_filters[prev_layer]
        
        if (conv.in_channels - len(prev_layer_filters)) == 0 and conv.out_channels - len(curr_layer_filters):
            print("Fully prunable layer")
        
        new_conv = \
                torch.nn.Conv2d(in_channels=conv.in_channels - len(prev_layer_filters),
                    out_channels=conv.out_channels - len(curr_layer_filters),
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding,
                    dilation=conv.dilation,
                    groups=conv.groups,
                    bias=(conv.bias is not None))
        
        #print(conv.bias)
        
        old_weights = conv.weight.data.cpu().numpy()
        new_weights = np.delete(old_weights, curr_layer_filters, axis=0)
        new_conv.weight.data = torch.from_numpy(new_weights)
        
        old_bias = conv.bias.data.cpu().numpy()
        new_bias = np.delete(old_bias, curr_layer_filters)
        new_conv.bias.data = torch.from_numpy(new_bias)
        
        if prev_layer != -1:
            old_weights = new_conv.weight.data.cpu().numpy()
            new_weights = np.delete(old_weights, prev_layer_filters, axis=1)
            new_conv.weight.data = torch.from_numpy(new_weights)
            
        #print(curr_layer, new_conv, last_layer)
        self.model.layers[curr_layer].conv = new_conv
        
        if last_layer:
            old_linear = self.model.fc
            
            new_linear = \
                torch.nn.Linear(new_conv.out_channels,
                               old_linear.out_features,
                               bias = (old_linear.bias is not None))
            
            old_weights = old_linear.weight.data.cpu().numpy()
            new_weights = np.delete(old_weights, curr_layer_filters, axis=1)
            
            new_linear.weight.data = torch.from_numpy(new_weights)
            new_linear.bias.data = old_linear.bias.data
            
            self.model.fc = new_linear