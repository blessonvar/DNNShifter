# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import typing
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import math
from math import log2



def vectorize(state_dict: typing.Dict[str, torch.Tensor]):
    """Convert a state dict into a single column Tensor in a repeatable way."""

    return torch.cat([state_dict[k].reshape(-1) for k in sorted(state_dict.keys())])


def unvectorize(vector: torch.Tensor, reference_state_dict: typing.Dict[str, torch.Tensor]):
    """Convert a vector back into a state dict with the same shapes as reference state_dict."""

    if len(vector.shape) > 1: raise ValueError('vector has more than one dimension.')

    state_dict = {}
    for k in sorted(reference_state_dict.keys()):
        if vector.nelement() == 0: raise ValueError('Ran out of values.')

        size, shape = reference_state_dict[k].nelement(), reference_state_dict[k].shape
        this, vector = vector[:size], vector[size:]
        state_dict[k] = this.reshape(shape)

    if vector.nelement() > 0: raise ValueError('Excess values.')
    return state_dict


def perm(N, seed: int = None):
    """Generate a tensor with the numbers 0 through N-1 ordered randomly."""

    gen = torch.Generator()
    if seed is not None: gen.manual_seed(seed)
    perm = torch.normal(torch.zeros(N), torch.ones(N), generator=gen)
    return torch.argsort(perm)


def shuffle_tensor(tensor: torch.Tensor, seed: int = None):
    """Randomly shuffle the elements of a tensor."""

    shape = tensor.shape
    return tensor.reshape(-1)[perm(tensor.nelement(), seed=seed)].reshape(shape)


def shuffle_state_dict(state_dict: typing.Dict[str, torch.Tensor], seed: int = None):
    """Randomly shuffle each of the tensors in a state_dict."""

    output = {}
    for i, k in enumerate(sorted(state_dict.keys())):
        output[k] = shuffle_tensor(state_dict[k], seed=None if seed is None else seed+i)
    return output

def prune(tensor: torch.Tensor, prune_fraction: float, min = True, mask: torch.Tensor = None):
    """Prune the remaining prune_fraction weights from the mak based on the scores in tensor"""
    if mask is None: mask = torch.ones_like(tensor)
    num_to_prune = np.ceil(torch.sum(mask).item() * prune_fraction).astype(int)
    sorted_tensor = torch.sort(tensor[mask == 1].reshape(-1)).values

    print ("num to prune", num_to_prune)
    if min:
       threshold = sorted_tensor[num_to_prune].double()
       return torch.where(tensor.double() > threshold, mask.double(), torch.zeros_like(tensor).double())
    else:
       threshold = sorted_tensor[len(sorted_tensor) - num_to_prune]
       return torch.where(tensor.double() < threshold, mask.double(), torch.zeros_like(tensor).double()).int()


def plot_distribution_weights(model, strategy, mask, prune_iterations, reinitialize, randomize_layerwise, result_folder):
    
    # create a PdfPages object
    file_name = strategy.capitalize() +"_Pruning"

    if reinitialize:
        file_name+= "_Reinit"
    if randomize_layerwise:
        file_name+= "_Randomize_layerwise"

    pdf = PdfPages(result_folder+file_name+"_Weights.pdf")

    f = open(result_folder+ "Plot_Details.txt", "a")
    f.write("\n\n"+file_name)

    for name, param in model.named_parameters(): 
        if name in mask:
            layer_score = torch.flatten(param.data).data.numpy()
            mask_data = torch.flatten(mask[name]).data.numpy()
            mask_percent = "%.2f" % (((len(mask_data)-sum(mask_data))/len(mask_data))*100)
            print ("Mask Weights %", mask_percent)

            updated_scores = layer_score*mask_data
            w1 = np.count_nonzero(layer_score)
            w2 = np.count_nonzero(updated_scores)

            title = "\n"+strategy.capitalize() +" Pruning | "+name+ "\n"+ "Prune %: "+ str(mask_percent) + " | Prune Iterations: "+ str(prune_iterations) +"\n" \
                "Weights Before Pruning: "+ str(w1) +" | Weights Removed: "+ str(w1-w2)+" | Weights After Pruning: "+ str(w2) +"\n" \
                    "Reinitialize: " + str(reinitialize) +" | Randomize Layerwise: "+ str(randomize_layerwise)
            
            bins = 100
            fig = plt.figure()
            plt.xlabel("Weights")
            plt.ylabel("Frequency")
            plt.hist([layer_score, updated_scores], bins, label=['Before Pruning', 'After Pruning'], color=["red","blue"])
            plt.legend(loc='upper right')
            fig.suptitle(title)
            fig.set_figheight(10)
            fig.set_figwidth(10)
            pdf.savefig(fig)

            # plt.show()
    f.close()
    pdf.close()
        
        
def plot_distribution_scores(scores, strategy, mask, prune_iterations, reinitialize, randomize_layerwise, result_folder):
    
    # create a PdfPages object
    file_name = strategy.capitalize() +"_Pruning"

    if reinitialize:
        file_name+= "_Reinit"
    if randomize_layerwise:
        file_name+= "_Randomize_layerwise"

    pdf = PdfPages(result_folder+file_name+"_Scores.pdf")

    for name, param in scores.items(): 
        original_score = torch.flatten(param).data.numpy()
        mask_data = torch.flatten(mask[name]).data.numpy()
        mask_percent = "%.2f" % (((len(mask_data)-sum(mask_data))/len(mask_data))*100)
        print ("Mask Scores %", mask_percent)

        # updated_scores = original_score*mask_data
        updated_scores = [original_score[i] for i in range(len(mask_data)) if mask_data[i]==1]

        w1 = np.count_nonzero(original_score)
        w2 = np.count_nonzero(updated_scores)

        title = "\n"+strategy.capitalize() +" Pruning | "+name+ "\n"+ "Prune %: "+ str(mask_percent) + " | Prune Iterations: "+ str(prune_iterations) +"\n" \
            "Weights Before Pruning: "+ str(w1) +" | Weights Removed: "+ str(w1-w2)+" | Weights After Pruning: "+ str(w2)+"\n" \
            "Reinitialize: " + str(reinitialize) +" | Randomize Layerwise: "+ str(randomize_layerwise)
        
        bins = 100
        fig = plt.figure()
        plt.xlabel("Scores")
        plt.ylabel("Frequency")
        plt.hist([original_score, updated_scores], bins, label=['Before Pruning', 'After Pruning'], color=["red","blue"])
        plt.legend(loc='upper right')
        fig.suptitle(title)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        pdf.savefig(fig)

    # plt.show()
    # remember to close the object to ensure writing multiple plots
    pdf.close()

def plot_distribution_scatter(scores, model, strategy, mask, prune_iterations, reinitialize, randomize_layerwise, result_folder):
    
    # create a PdfPages object
    file_name = strategy.capitalize() +"_Pruning"
    if reinitialize:
        file_name+= "_Reinit"
    if randomize_layerwise:
        file_name+= "_Randomize_layerwise"

    pruning_type = strategy.capitalize() +" Pruning"
    pdf = PdfPages(result_folder+file_name+"_Weights_Scatter.pdf")

    f = open(result_folder+ "Plot_Details.txt", "a")
    f.write("\n\n"+pruning_type)

    for name, param in model.named_parameters(): 
        if name in mask:
            weight_layer = torch.flatten(param.data).data.numpy()
            score_layer = torch.flatten(scores[name]).data.numpy()

            mask_data = torch.flatten(mask[name]).data.numpy()
            mask_percent = "%.2f" % (((len(mask_data)-sum(mask_data))/len(mask_data))*100)

            # Which indexes, weights and scores will be pruned
            masking_indexes = [i for i in range(len(mask_data)) if mask_data[i]==0]
            mask_weights = [weight_layer[i] for i in masking_indexes]
            mask_scores = [score_layer[i] for i in masking_indexes]

            # Which indexes, weights and scores will remain
            rem_indexes = [i for i in range(len(mask_data)) if mask_data[i]!=0]
            rem_weights = [weight_layer[i] for i in rem_indexes]
            rem_scores = [score_layer[i] for i in rem_indexes]

            fig = plt.figure()
            plt.xlabel("Scores")
            plt.ylabel("Weights")

            if reinitialize or randomize_layerwise:
                plt.scatter(rem_scores, rem_weights, alpha=0.05, c='b')
                plt.scatter(mask_scores, mask_weights, alpha=0.05, c='r')

            else:
                plt.scatter(score_layer, weight_layer, alpha=0.05)
                plt.axvline(x=max(mask_scores), c='r')

            title = "\n"+strategy.capitalize() +" Pruning | "+name+ "\n"+ "Prune %: "+ str(mask_percent) + " | Prune Iterations: "+ str(prune_iterations) +"\n" \
                "Mask %: " + mask_percent +" | Reinitialize: " + str(reinitialize) +" | Randomize Layerwise: "+ str(randomize_layerwise)

            fig.suptitle(title)
            fig.set_figheight(10)
            fig.set_figwidth(10)
            pdf.savefig(fig)

    # plt.show()
    f.close()
    pdf.close()


def weights_analysis(model, strategy, reinitialize, randomize_layerwise, status):
    
    # create a PdfPages object
    file_name = strategy.capitalize() +"_Pruning_"

    if reinitialize:
        file_name+= "_Reinit_"
    if randomize_layerwise:
        file_name+= "_Randomize_layerwise_"
           
    torch.save(model.state_dict(), file_name+status+'.pt')
    # torch.save(model, file_name+'_model.pt')


# calculate the kl divergence
def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))