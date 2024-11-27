import sys
import os
from paths import *
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
from collections import OrderedDict
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)
from Wormholes.wormholes.model_utils import make_and_restore_model
from Wormholes.wormholes.datasets import DATASETS

class SkipConnection(nn.Module):
    def __init__(self, layer_type,in_features, in_channels, out_channels, stride, device):
        super(SkipConnection, self).__init__()
        self.layer_type = layer_type
        self.device = device
        self.out_features = 4096
        self.in_features = in_features
        if self.layer_type == 'Conv2d':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        elif self.layer_type == 'Linear':
            self.linear = nn.Linear(in_features=self.in_features,out_features=self.out_features, device=self.device, bias=False)

    def forward(self, x):
        if self.layer_type == 'Conv2d':
            x = self.conv(x)
            return x
        elif self.layer_type == 'Linear':
            x = self.linear(x)
            return x

class Model:
    def __init__ (self,model_name,pretrained,remove_fc,skip_connections_fc,skip_connections_conv2d, layers_to_remove, arch_type, input_size=9216, output_size=1, device = 'cuda', transform = None, download_model = True, robust_model = False, robustness_level = '3', gerious_models = False):
        self.model_name = model_name
        self.pretrained = pretrained
        self.output_size = output_size
        self.transform = transform
        self.remove_fc = remove_fc
        self.device = device
        self.skip_connections_fc = skip_connections_fc 
        self.skip_connections_conv2d = skip_connections_conv2d 
        self.add_skip_connections = None 
        self.activations = {}
        self.model = None 
        self.layers_to_remove = layers_to_remove
        self.arch_type = arch_type
        self.input_size = input_size
        self.download_model = download_model
        self.robust_model = robust_model
        self.robustness_level = robustness_level

        self.gerious_models = gerious_models


    def load_gerious_models(self):

        model = getattr(models, 'resnet50')(weights=None)
        checkpoint = torch.load(f'{CHECKPOINTS}/geirhos/resnet50_l2_eps{self.robustness_level}.ckpt',map_location='cpu')
        sd = {k[len('module.model.'):]:v for k,v in checkpoint['model'].items()\
              if k[:len('module.model.')] == 'module.model.'}  
        model.load_state_dict(sd)
        print('Loaded gerious models')

        return model
    
    def load(self):
        if self.robust_model:
            dataset = DATASETS['imagenet']('data_path')	
            parent_model = make_and_restore_model(arch=self.model_name,dataset=dataset,pytorch_pretrained=True, resume_path=f'{parent_dir}/Wormholes/scripts/download/checkpoints/imagenet_l2_{self.robustness_level}_0.pt')
            
            model = parent_model[0].model
        elif self.gerious_models:
            model = self.load_gerious_models()
        else:
            if self.download_model:
                if self.model_name in models.list_models():
                    if self.pretrained == True:
                        print("load from Pytorch torchvision Module")
                        model = getattr(models, self.model_name)(weights='DEFAULT')

                    else:
                        print('loading untrained model')
                        model = getattr(models, self.model_name)(weights=None)
                else:
                    raise ValueError(f"{self.model_name} is not available in torchvision.models.")
            else:
                if self.pretrained == True:

                    model = getattr(models, self.model_name)(weights=None)
                    checkpoint = torch.load(f'{CHECKPOINTS}/{self.model_name}/{self.model_name}.pth')
                    model.load_state_dict(checkpoint) 
                else:
                    model = getattr(models, self.model_name)(weights=None)
                    print('LOADING UNTRAINED MODEL')

                
        if self.skip_connections_fc:
                model = self.add_skip_connections_linear(model)
                model.forward = self.forward_skip

        if self.skip_connections_conv2d:
                model = self.add_skip_connections_downsample(model)
                model.forward = self.forward_skip

        if self.remove_fc:
                model = self.remove_layer(model)

        self.model = model.to(self.device)
            
        return model
    
    def configure_layers(self):
        if self.arch_type !=None:
            correct_layers = [f'{self.arch_type}.{layer}' for layer in self.layers_to_remove]
        else:
            correct_layers = self.layers_to_remove
        return correct_layers
               
              
    def remove_layer(self,model):
        corrected_layers = self.configure_layers()
        if self.arch_type ==None:
            for layers in corrected_layers:
                delattr(model, layers)
            model.add_module('fc', nn.Linear(self.input_size, self.output_size))
        else:
            for layer in corrected_layers:
                parent_module, child_module = layer.split('.')
                if hasattr(model, parent_module):
                    if hasattr(getattr(model, parent_module), child_module):
                        delattr(getattr(model, parent_module), child_module)
                    else:
                        print(f"Module {child_module} not found in {parent_module}")
                else:
                    print(f"Module {parent_module} not found in model")
            parent_module = getattr(model, self.arch_type)
            parent_module.add_module('fc', nn.Linear(self.input_size, self.output_size))
        return model
    
    def hook_fn(self,name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook

    def add_skip_connections_downsample(self, model):
        layers = []
        for name, layer in model.features.named_children():
            if isinstance(layer, nn.Conv2d):
                outplanes = layer.out_channels
                inplanes = layer.in_channels
                stride = 1 
                skip_connection = SkipConnection('Conv2d',in_channels=outplanes, out_channels=outplanes, stride=stride, device=self.device)
                layers.extend([layer, nn.ReLU(inplace=True), skip_connection])
            else:
                layers.append(layer)  
        model.features = nn.Sequential(*layers)
        #self._save_hook(model)

        return model
    
    def add_skip_connections_linear(self, model):
        
        model.classifier.add_module('skip1', SkipConnection('Linear', in_features=484992,in_channels=None, out_channels=None, stride=None, device=self.device))
        return model
    

    def forward_skip(self, x):
        
        x_skip = torch.zeros(x.size(0), 0, device=x.device)

        skip_indices = {0, 3, 6, 8, 10}

        for i, layer in enumerate(self.model.features):
            x = layer(x)

            if i in skip_indices:
                x_flattened = x.view(x.size(0), -1)
                x_skip = torch.cat((x_skip, x_flattened), dim=1)

        x_skip = self.model.classifier[4](x_skip)
        out = self.model.classifier[5](x_skip)

        return out