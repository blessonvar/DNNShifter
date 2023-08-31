import sys
import torch
import time
sys.path.append("../")
import datasets
from models import registry

class ModelProfiler:
    def __init__(self, dataset):
        default_hparams = None
        if dataset == 'CIFAR10':
            default_hparams = registry.get_default_hparams('cifar_vgg_16')
        elif dataset == 'TinyImageNet':
            default_hparams = registry.get_default_hparams('tinyimagenet_resnet_50')
        
        self.testloader = datasets.registry.get(default_hparams.dataset_hparams, train=False)
        
    
    def test(self, model):
        device = 'cuda'
        example_count = torch.tensor(0.0).to(device)
        total_correct = torch.tensor(0.0).to(device)
        
        model.eval()
        model.cuda()
    
        with torch.no_grad():
            for examples, labels in self.testloader:
                examples = examples.to(device)
                labels = labels.squeeze().to(device)
               
                output = model(examples)

                labels_size = torch.tensor(len(labels), device=device)
                example_count += labels_size
                total_correct += torch.sum(torch.eq(labels, output.argmax(dim=1)))
                
                del examples
                del labels
                del output
            
        total_correct = total_correct.cpu().item()
        example_count = example_count.cpu().item()
    
        return total_correct / example_count
    
    def params(self, model):
        nz = sum(p.nonzero().size(0) for p in model.parameters() if p.requires_grad)
        p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return nz, p
    
    def infer(self, device, model):        
        total_inf_start = time.time()
        total_examples = 0
        
        model.eval()
        inf_list = []
        
        
        _warmup_threshold = 7
        if device == "cpu":
            model.cpu()
            max_itr = 2 + _warmup_threshold
        elif device == "cuda":
            model.cuda()
            max_itr = 25 + _warmup_threshold
        
        itr = 0
        with torch.no_grad():
            for examples, _ in self.testloader:
                if itr < max_itr:
                    examples = examples.to(device)
                    total_examples = total_examples + len(examples)
            
                    inf_start = time.time()
                    model(examples)
                    
                    if device == "cuda":
                        torch.cuda.synchronize()
            
                    inf_end = time.time()
            
                    inf_list.append((inf_end - inf_start) * 1000)
                
                    del examples
                    itr = itr + 1
                
                
        total_inf_end = time.time()
        inf_list = inf_list[_warmup_threshold:] # Remove timings affected by device warmup
        inf_list = inf_list[:-1] # Remove last timing incase test size is not fully divisble by batch size
        #return format(((total_inf_end - total_inf_start) *1000) / total_examples, '.4f')
        return format(((sum(inf_list) / len(inf_list)) *1000) / total_examples, '.4f')
    
    def profile(self, model, no_cpu=False, no_acc_test=False):
        results = {}
        results["# Non-Zero Params"], results["# Params"] = self.params(model)
        
        if not no_acc_test:
            results["Top-1 Accuracy"] = format(self.test(model) * 100, '.2f')
        
        #warm-up
        tmp1 = self.infer("cuda", model)
        #tmp2 = self.infer("cpu", model)
        
        if not no_cpu:
            results["CPU Inf (ms)"] = self.infer("cpu", model)
        results["GPU Inf (ms)"] = self.infer("cuda", model)
        
        return results
    