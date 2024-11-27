import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

import time

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from sklearn.decomposition import PCA

import torchvision
from stimupy.noises.whites import white
import matplotlib.pyplot as plt
import os
from torchvision.models.alexnet import alexnet,AlexNet_Weights
from torch.utils.data import Dataset,DataLoader
from paths import *
import numpy as np
from torchvision.utils import make_grid
from models.model_arch import Model
from models.model_arch import Model
from database.coco import NSD,COCO,create_transform,create_transform_aperature,get_data_loaders,generate_test_transforms
from utilities.plot_utils import (plot_si_vs_rdm,
                                  train_histogram,
                                  test_histogram,
                                  plot_kernel_correlations,
                                  fit_psychometric_curve,
                                  plot_overlapping_histograms,
                                  plot_mean_std_scatter,
                                  plot_mean_std_vs_SI)
from utilities.train_utils import (two_alt_forced_choise,
                                   compute_rdm,
                                   calculate_dpime,
                                   calculate_specificity_dprime,
                                   find_top_pairs,
                                   find_extereme_pairs,
                                   find_random_pairs,
                                   freeze_layers_except_last,
                                   rdm_decompostion,
                                   retrieve_dissimilarities,
                                   process_test_data,
                                   calculate_specificity_loss,
                                   get_top_kernels_for_pair,
                                   get_top_values_for_pair,
                                   )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_time = time.time()

activation = {}
hook_handles = {}  # Stores hook handles

def get_activation_list(model, dataloader,device):
    all_activation = {}
    for i, images in enumerate(dataloader):
        print(f'processed {i+1} batches')
        _ = model(images[0].to(device))
        if len(all_activation) == 0:
            for k in activation.keys():
                all_activation[k] = activation[k].cpu().numpy()
        else: 
            for k in activation.keys():
                all_activation[k] = np.concatenate([all_activation[k],activation[k].cpu().numpy()])                
    return all_activation

def save_act(model,layer_names):
    global hook_handles  # Use the global dictionary to store hook handles
    for name, module in model.named_modules():
        if name in layer_names:
            # Register hook and store its handle
            handle = module.register_forward_hook(get_activation(name, activation))
            hook_handles[name] = handle
def get_layer_names(model):
    layer_names = []
    for name,layer in model.named_modules():
        if isinstance(layer,(nn.Conv2d,nn.Linear,nn.AdaptiveAvgPool2d)):
            layer_names.append(name)
    layer_names = layer_names[:-1]
    return layer_names

def get_activation(name, activation):        
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
def remove_hooks():
    """
    Removes all registered hooks from the model.
    """
    global hook_handles
    for handle in hook_handles.values():
        handle.remove()
    hook_handles.clear()  # C


def retrieve_PCA(pca_train,activation_test, pc1=0):

    pca_result_test= pca_train.transform(activation_test)

    return pca_result_test[:, pc1]

def fit_PCA(activation_train):

    pca = PCA(n_components=2)

    pca_result_test= pca.fit(activation_train)

    return pca_result_test
        
model_name = 'resnet101'
net = Model(model_name,pretrained=True,remove_fc=True,skip_connections_conv2d=False,skip_connections_fc=False,layers_to_remove=['fc'], arch_type = None,input_size=2048,output_size=1,download_model=False, device=device,robust_model=False,robustness_level='1').load().train()

locations = {(0,0): {}, (0,120): {}, (120,0): {}, (120,120):{}}
#locations = {(0,0): {}}



train_loc = {}

print(locations)

configs_tasks_train= [
    {
        'noise_type': 'None', 
        'intensity_range': (0,0.5), 
        'type': 'circular_gaussian', 
        'radius': None, 
        'bg_size': (350, 350), 
        'position': (60, 60), 
        'zoom_factor': 1,
        'rotation': None,
        'horizontal_flip': None
    },
]


model = net
layer_to_save = 'avgpool'
save_act(model,layer_names=layer_to_save)

train_task_config = configs_tasks_train[0]
IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN = generate_test_transforms(train_task_config)
dataset_train = NSD(data_dir=NSD_DIR,transforms=IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN)
images_dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=False)
model_activation = get_activation_list(model,images_dataloader_train,device=device)
my_act_train = torch.tensor(model_activation[layer_to_save]).reshape(1000,-1).numpy()

pca_model = fit_PCA(my_act_train)
pca_result_train = pca_model.transform(my_act_train)[:,0]
train_loc = pca_result_train

remove_hooks()


for location in locations:
    print(location)
    configs_tasks_pca= [
        {
            'noise_type':'None', 
            'intensity_range': (0,0.5), 
            'type': 'circular_gaussian', 
            'radius': None, 
            'bg_size': (350, 350), 
            'position': location, 
            'zoom_factor': 1,
            'rotation': None,
            'horizontal_flip': False
        },
    ]
    save_act(model,layer_names=layer_to_save)

    IMAGENET_TRANSFORM_NOISE_APRATUR_PCA = generate_test_transforms(configs_tasks_pca[0])
    dataset_test = NSD(data_dir=NSD_DIR,transforms=IMAGENET_TRANSFORM_NOISE_APRATUR_PCA)
    images_dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
    model_activation_test = get_activation_list(model,images_dataloader_test,device=device)

    remove_hooks()
    layer_to_save = 'avgpool'


    my_act_test = torch.tensor(model_activation_test[layer_to_save]).reshape(1000,-1).numpy()


    pca_result_test = pca_model.transform(my_act_test)[:,0]


    locations[location] = pca_result_test - train_loc
    #locations[location] = pca_result_test

locations_train = {}

locations_train['Train']=train_loc


plt.figure(figsize=(14, 8))
plt.title(f'PCA 1 values for {model_name}')
colors = plt.cm.viridis(np.linspace(0, 1, 1000))

x_ticks = ['Train'] + [f'Location {loc}' for loc in sorted(locations.keys())]
n_locations = len(locations) +1 


for loc_index, (location, values) in enumerate(sorted(locations_train.items()), start=0):
    for value_index, value in enumerate(values):
        plt.scatter(loc_index, value, color=colors[value_index], zorder=2)

for loc_index, (location, values) in enumerate(sorted(locations.items()), start=1):
    for value_index, value in enumerate(values):
        plt.scatter(loc_index, value, color=colors[value_index], zorder=2)

plt.xticks(range(n_locations), labels=x_ticks)
plt.ylabel('Difference PCA values from training')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.tight_layout()
path = os.path.join(f'{FIGURE_DIR}/location_invariant',f'{model_name}')
plt.savefig(f'{path}/run_PCA_values_{model_name}.pdf',format='pdf')
#plt.show()



plt.figure(figsize=(14, 8))
plt.title(f'PCA 1 values for {model_name}')
colors = plt.cm.viridis(np.linspace(0, 1, 1000))

x_ticks = ['PC Train Value'] + [f'Location {loc}' for loc in sorted(locations.keys())]
n_locations = len(locations) + 1

train_means = [np.mean(values) for location, values in sorted(locations_train.items())]
train_stds = [np.std(values) for location, values in sorted(locations_train.items())]
plt.errorbar(0, train_means, yerr=train_stds, fmt='o', color=colors[0], zorder=2,capsize=4)

for loc_index, (location, values) in enumerate(sorted(locations.items()), start=1):
    mean = np.mean(values)
    std = np.std(values)
    plt.errorbar(loc_index, mean, yerr=std, fmt='o', color=colors[loc_index+100], zorder=2,capsize=4)

plt.xticks(range(n_locations), labels=x_ticks)
plt.ylabel('Mean difference PCA values from training')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.grid(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.tight_layout()
path = os.path.join(f'{FIGURE_DIR}/location_invariant',f'{model_name}')
plt.savefig(f'{path}/mean_PCA_values_{model_name}.pdf',format='pdf')

#plt.savefig(f'/Users/amirozhandehghani/Desktop/run/mean_PCA_values_{model_name}.pdf', format='pdf')
#plt.show()


path = os.path.join(f'{FIGURE_DIR}/location_invariant',f'{model_name}')
save_file = os.makedirs(name=path,exist_ok=True)
plt.savefig(f'{path}/PCA_values_{model_name}.pdf',format='pdf')

print('time took:', time.time()-start_time)
