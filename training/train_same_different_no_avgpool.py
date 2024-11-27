import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import json

from database.coco import NSD,COCO,create_transform,create_transform_aperature,get_data_loaders,generate_test_transforms
import argparse
from paths import *
from paths import  FIGURE_DIR
import copy
import time
from models.model_arch import Model
from utilities.extract_activations import get_activation_list,save_act,get_layer_names
from utilities.plot_utils import (
                                  plot_si_vs_rdm,
                                  train_histogram,
                                  plot_kernel_correlations,
                                  fit_psychometric_curve,
                                  plot_overlapping_histograms,
                                  plot_mean_std_scatter,
                                  plot_mean_std_vs_SI,
                                  plot_all_pairs_losses,
                                  save_to_json,
                                  plot_test_losses_for_specific_test,
                                  
                                  )

from utilities.train_utils import (two_alt_forced_choise,
                                   compute_rdm,
                                   find_top_pairs,
                                   find_low_pairs,
                                   find_extereme_pairs,
                                   calculate_dpime,
                                   find_random_pairs,
                                   
                                   freeze_layers_except_last,
                                   rdm_decompostion,
                                   retrieve_dissimilarities,
                                   process_test_data,
                                   calculate_specificity_loss,
                                   get_top_kernels_for_pair,
                                   get_top_values_for_pair,
                                   find_top_pairs_eigen_decomposition,
                                   compute_act_mat,
                                   rdm_decompostion_short,
                                   retrieve_eigenvectors_torch,
                                   find_extereme_pairs_across_kernels,)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='VPL training')
parser.add_argument('--arch', type=str, default='alexnet')
parser.add_argument('--name', type=str, default='test_1')
parser.add_argument('--skip_connections_fc', default=False, action='store_true')
parser.add_argument('--skip_connections_conv2d', default=False, action='store_true')
parser.add_argument('--freeze_weights', default=True, action='store_true')
parser.add_argument('--log_interval', type=int, default=2000)
parser.add_argument('--trainset', type=str, default='COCO')
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--remove_fc', action='store_true', default=True)
parser.add_argument('--pretrained',action='store_true' , default=True)
parser.add_argument('--noise_type', type=str, default='None')
parser.add_argument('--intensity_range', nargs='+', type=float)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--task_prob', type=float, default=0.5)
parser.add_argument('--num_pairs', type=int, default=20)
parser.add_argument('--iterations', type=int, default=50)
parser.add_argument('--input_size_fc', type=int, default=2048)
parser.add_argument('--RDM_layer', type=str, default='layer4.2.relu')
parser.add_argument('--download_model',default=False, action='store_true')
parser.add_argument('--extreme_pairs', default=False, action='store_true')
parser.add_argument('--top_kernel_indexes', type=int, default=1)
parser.add_argument('--kernel_decomposition', type=int, default=999)
parser.add_argument('--arch_type', type=str, default='None')
parser.add_argument('--layers_to_remove', type=str, default='fc')
parser.add_argument('--robust_model', default=False, action='store_true')
parser.add_argument('--robustness_level', type=str, default='3')
parser.add_argument('--gerious_models',default=False, action='store_true')

args = parser.parse_args()

start_time = time.time()

if args.arch_type == 'None':
    arch_type = None
else:
    arch_type = args.arch_type
layers_to_remove = []
if args.layers_to_remove == 'None':
    layers_to_remove = ['0','1','2','3','4','5','6']
else:
    layers_to_remove.append(args.layers_to_remove)

LOG_DIR = os.path.join(FIGURE_DIR,f'{args.name}_{args.arch}_{args.lr}_{args.num_pairs}_{args.iterations}_{args.noise_type}_{args.batchsize}_{args.extreme_pairs}_{args.robustness_level}') 
os.makedirs(LOG_DIR, exist_ok=True)
model = Model(model_name=args.arch,pretrained=args.pretrained,remove_fc=args.remove_fc,skip_connections_fc=args.skip_connections_fc,skip_connections_conv2d=args.skip_connections_conv2d,device=device,layers_to_remove=layers_to_remove, arch_type = arch_type,input_size=args.input_size_fc,output_size=1,download_model=args.download_model, robust_model = args.robust_model, robustness_level = args.robustness_level, gerious_models = args.gerious_models).load().eval().to(device)

model.avgpool = nn.Identity()
delattr(model,'fc')
model.add_module('fc',nn.Linear(100352,1))
model.fc.to(device)

print(model)

args_dict = vars(args)

# Save to JSON file
with open(f'{LOG_DIR}/config.json', 'w') as f:
    json.dump(args_dict, f, indent=4)

#save_args_to_file(args,f'{LOG_DIR}/ags.txt')

#writer = SummaryWriter(LOG_DIR)

#layer_names = get_layer_names(model)
layer_names = [args.RDM_layer]
save_act(model,layer_names=layer_names)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

print(device)
intensity_range = tuple(args.intensity_range)

configs_tasks= [
    {
        'noise_type': args.noise_type, 
        'intensity_range': intensity_range, 
        'contrast':0.2,
        'type': 'circular_gaussian', 
        'radius': None, 
        'bg_size': (350, 350), 
        'position': (60, 60), 
        'zoom_factor': 1,
        'rotation': None,
        'horizontal_flip': None
    },
    {
    'noise_type': args.noise_type, 
        'intensity_range': intensity_range, 
        'contrast':0.2,
        'type': 'circular_gaussian', 
        'radius': None, 
        'bg_size': (350, 350), 
        'position': (0, 0), 
        'zoom_factor': 1,
        'rotation': None,
        'horizontal_flip': None
    },
    {
    'noise_type': args.noise_type, 
            'intensity_range': intensity_range, 
            'contrast':0.2,
            'type': 'circular_gaussian', 
            'radius': None, 
            'bg_size': (350, 350), 
            'position': (60, 60), 
            'zoom_factor': 2,
            'rotation': None,
            'horizontal_flip': None
        },
    {
    'noise_type': args.noise_type, 
            'intensity_range': intensity_range, 
            'contrast':0.2,
            'type': 'circular_gaussian', 
            'radius': None, 
            'bg_size': (350, 350), 
            'position': (60, 60), 
            'zoom_factor': 4,
            'rotation': None,
            'horizontal_flip': None
        },
    {
    'noise_type': args.noise_type, 
            'intensity_range': intensity_range, 
            'contrast':0.2,
            'type': 'circular_gaussian', 
            'radius': None, 
            'bg_size': (350, 350), 
            'position': (60, 60), 
            'zoom_factor': 1,
            'rotation': 90,
            'vertical_flip': None
        },
]

train_task_config = configs_tasks[0]
IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN = generate_test_transforms(train_task_config)
dataset = NSD(data_dir=NSD_DIR,transforms=IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN)
images_dataloader = DataLoader(dataset, batch_size=5, shuffle=False,)
model_activation = get_activation_list(model,images_dataloader,device=device)
#import pdb;pdb.set_trace()
my_act = torch.tensor(model_activation[args.RDM_layer]).reshape(1000,-1)
#my_act = my_act[:500]
#rdm = compute_rdm(my_act)

rdm = compute_act_mat(my_act)

top_n = args.num_pairs
if args.extreme_pairs:
    #eigenvectors = rdm_decompostion_short(rdm)
    #kernels_torch = retrieve_eigenvectors_torch(eigenvectors)
    #top_100_pairs = find_extereme_pairs_across_kernels(kernels_torch,top_k=top_n)
    top_100_pairs = find_extereme_pairs(rdm,top_n=top_n)
    print('Extreme_pairs')
else: 
    print('None')
    top_100_pairs = find_random_pairs(rdm,top_n=top_n)
    print('RANDOM PAIRS')
    #top_100_pairs = find_top_pairs(rdm,top_n=top_n) 


print('top_n: ',top_n)
#print(top_100_pairs)
pairs_train = top_100_pairs
#pairs_train = [(483,40),(542,483),(483,172),(542,40),(36,6),(75,36)]
#pairs_train = find_top_pairs_eigen_decomposition(top_100_pairs,rdm)

def test(model,pairs_trains,name_test,config_test):
    with torch.no_grad():
        model.eval()
        criterion = nn.BCELoss()  
        IMAGENET_TRANSFORM_NOISE_APRATURE_TEST = generate_test_transforms(config_test)
        test_loader, _ = get_data_loaders(data_dir=COCO_DIR, transform=IMAGENET_TRANSFORM_NOISE_APRATURE_TEST, batch_size=args.batchsize, same_pair_probability=args.task_prob,same_not_rand=True, same_reference=False, ref_state=None, ref_other=None,idx_ref=pairs_trains[0],idx_other=pairs_trains[1])
        losses_dict = {}
        prob_list_test = []
        count = 0
        total_loss = 0

        for batch_idx, (img1_batch, img2_batch, label1_batch, label2_batch, same_pair) in enumerate(test_loader):
            if batch_idx == 1 or batch_idx == 2:
                img1_batch = img1_batch.to(device)
                img2_batch = img2_batch.to(device)
                label1_batch = label1_batch.to(device)
                label2_batch = label2_batch.to(device)

                output_1 = model(img1_batch)
                output_2 = model(img2_batch)

                _, prob = two_alt_forced_choise(output_1, output_2)

                prob_list_test.append(prob)

                TRUE_LABEL = torch.where(same_pair, torch.ones_like(same_pair), torch.zeros_like(same_pair))
                TRUE_LABEL = TRUE_LABEL.float().unsqueeze(1).to(device)

                test_dprime = calculate_dpime(prob=prob.detach().cpu(),TRUE_lABEL=TRUE_LABEL.detach().cpu())

                loss = criterion(prob, TRUE_LABEL)

                total_loss = total_loss + loss.item()
                count += 1

                if batch_idx == 2:
                    break

        if count > 0:
            average_loss = total_loss / count
            print('Average Loss: ', average_loss)
            return average_loss,test_dprime
        else:
            return None, None, None 

def register_test_loss(model, pairs_trains, configs_tasks, iteration, test_loss_per_iteration_dict,test_number=1):

    #i=1
    i = test_number

    test_name = f'test{i}'
    
    test_loss,_ = test(model=model, pairs_trains=pairs_trains, name_test=test_name, config_test=configs_tasks[i])
    
    pair_key = tuple(pairs_trains)
    
    if pair_key not in test_loss_per_iteration_dict:
        test_loss_per_iteration_dict[pair_key] = {}
    
    if test_name not in test_loss_per_iteration_dict[pair_key]:
        test_loss_per_iteration_dict[pair_key][test_name] = {}
    
    test_loss_per_iteration_dict[pair_key][test_name][iteration] = test_loss

    return test_loss_per_iteration_dict

def prob_to_classification(prob):

    label = torch.where(prob >=0.5, torch.ones_like(prob), torch.zeros_like(prob))

    return label

def train(pairs_train,config_train,iterations,test_number=1):
    loss_dict = {}
    loss_test_dict = {}
    test_loss_per_iteration_dict = {}

    SI_dict = {}   
    SI_dict_dprime = {}    

    for pairs_trains in pairs_train:
        
        IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN = generate_test_transforms(config_train)

        #model = Model(model_name=args.arch,pretrained=args.pretrained,remove_fc=args.remove_fc,skip_connections_fc=args.skip_connections_fc,skip_connections_conv2d=args.skip_connections_conv2d,device=device,layers_to_remove=layers_to_remove, arch_type = arch_type,input_size=args.input_size_fc,output_size=1,download_model=args.download_model)
        model = Model(model_name=args.arch,pretrained=args.pretrained,remove_fc=args.remove_fc,skip_connections_fc=args.skip_connections_fc,skip_connections_conv2d=args.skip_connections_conv2d,device=device,layers_to_remove=layers_to_remove, arch_type = arch_type,input_size=args.input_size_fc,output_size=1,download_model=args.download_model, robust_model = args.robust_model, robustness_level = args.robustness_level, gerious_models = args.gerious_models).load().train()

        model.avgpool = nn.Identity()
        delattr(model,'fc')
        model.add_module('fc',nn.Linear(100352,1))
        
        model.fc.weight = torch.nn.Parameter(torch.zeros(model.fc.weight.shape))

        model.fc.bias = torch.nn.Parameter(torch.zeros(model.fc.bias.shape))

        model.fc.to(device)

        criterion = nn.BCELoss()  

        freeze_layers_except_last(model,last_layer_name='fc')

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9) 
        
        train_loader, _ = get_data_loaders(data_dir=COCO_DIR, transform=IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN, batch_size=args.batchsize, same_pair_probability=args.task_prob,same_not_rand=True, same_reference=False, ref_state=None, ref_other=None,idx_ref=pairs_trains[0],idx_other=pairs_trains[1])

        lossess = []

        prob_list_train = []
        train_dprime_list = []
        correct = 0
        total = 0
        print(f'noise_type: {args.noise_type}')
        for batch_idx, (img1_batch, img2_batch, label1_batch, label2_batch, same_pair) in enumerate(train_loader):
            
            while len(lossess) < iterations: 
                #model.train()
                #freeze_layers_except_last(model,last_layer_name='fc')

                iteration = len(lossess)
                img1_batch = img1_batch.to(device)
                img2_batch = img2_batch.to(device)
                label1_batch = label1_batch.to(device)
                label2_batch = label2_batch.to(device)

                output_1 = model(img1_batch)
                output_2 = model(img2_batch)

                delta_h, prob = two_alt_forced_choise(output_1, output_2)

                prob_list_train.append(prob)

                label = prob_to_classification(prob)                

                TRUE_LABEL = torch.where(same_pair, torch.ones_like(same_pair), torch.zeros_like(same_pair))
                TRUE_LABEL = TRUE_LABEL.float().unsqueeze(1).to(device)
                
                correct += (label == TRUE_LABEL).sum().item()

                total += TRUE_LABEL.size(0)

                print(f'Accuracy of the network train images: {100 * correct // total} %')
                loss = criterion(prob, TRUE_LABEL)

                lossess.append(loss.item())

                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()

                print('loss: ', loss)

                #model_copy = copy.deepcopy(model)
                
                test_loss_per_iteration_dict = register_test_loss(model, pairs_trains, configs_tasks, iteration, test_loss_per_iteration_dict,test_number=test_number)


                train_dprime = 0
                #train_dprime = calculate_dpime(prob=prob.detach().cpu(),TRUE_lABEL=TRUE_LABEL.detach().cpu())

                train_dprime_list.append(train_dprime)

                #train_histogram(noise_type=args.noise_type,intensity_range=intensity_range,porb=porb,TRUE_LABEL=TRUE_LABEL,step=i,pairs=pairs_trains)
        average_loss,_ = test(model=model,pairs_trains=pairs_trains,name_test='test1',config_test=configs_tasks[test_number])
        loss_test_dict[tuple(pairs_trains), 'test1']=average_loss
        average_loss_2,_ = test(model=model,pairs_trains=pairs_trains,name_test='test2',config_test=configs_tasks[2])
        average_loss_3,_ = test(model=model,pairs_trains=pairs_trains,name_test='test3',config_test=configs_tasks[3])
        average_loss_4,_ = test(model=model,pairs_trains=pairs_trains,name_test='test4',config_test=configs_tasks[4])
        loss_test_dict[tuple(pairs_trains), 'test2']=average_loss_2
        loss_test_dict[tuple(pairs_trains), 'test3']=average_loss_3
        loss_test_dict[tuple(pairs_trains), 'test4']=average_loss_4

        #SI_measure =  (test_dprime - train_dprime_list[-1]) / (train_dprime_list[0] - train_dprime_list[-1])
        SI_measure = 0
        #SI_dict_dprime[tuple(pairs_trains), 'test1'] = SI_measure
        #loss_test_dict[tuple(pairs_trains), 'test1']=test(model=model,pairs_trains=pairs_trains,name_test='test1',config_test=configs_tasks[1])
        #loss_test_dict[tuple(pairs_trains), 'test2']=test(model=model,pairs_trains=pairs_trains,name_test='test2',config_test=configs_tasks[2])
        #loss_test_dict[tuple(pairs_trains), 'test3']=test(model=model,pairs_trains=pairs_trains,name_test='test3',config_test=configs_tasks[3])
        #loss_test_dict[tuple(pairs_trains), 'test4']=test(model=model,pairs_trains=pairs_trains,name_test='test4',config_test=configs_tasks[4])
        loss_dict[(pairs_trains, (0,1))] = lossess

        SI_dict[tuple(pairs_trains), 'test1'] = calculate_specificity_loss(test=loss_test_dict[tuple(pairs_trains), 'test1'],train=loss_dict[(pairs_trains, (0,1))])
        SI_dict[tuple(pairs_trains), 'test2'] = calculate_specificity_loss(test=loss_test_dict[tuple(pairs_trains), 'test2'],train=loss_dict[(pairs_trains, (0,1))])
        SI_dict[tuple(pairs_trains), 'test3'] = calculate_specificity_loss(test=loss_test_dict[tuple(pairs_trains), 'test3'],train=loss_dict[(pairs_trains, (0,1))])
        SI_dict[tuple(pairs_trains), 'test4'] = calculate_specificity_loss(test=loss_test_dict[tuple(pairs_trains), 'test4'],train=loss_dict[(pairs_trains, (0,1))])
    print('DONE')
    return SI_dict, loss_dict, test_loss_per_iteration_dict,SI_dict_dprime

iterations = args.iterations
SI_DICT,loss_dict, test_loss_per_iteration_dict,SI_dict_dprime  = train(pairs_train=pairs_train,config_train=configs_tasks[0],iterations=iterations)
save_to_json(loss_dict, SI_DICT, rdm,savedir=LOG_DIR)
plot_test_losses_for_specific_test(test_loss_per_iteration_dict, 'test1', to_save=True, save_json=True,savedir=LOG_DIR)
plot_all_pairs_losses(loss_dict,color_split=False,savedir = LOG_DIR)
kernels = [rdm_decompostion(kernel_num=i, rdm=rdm) for i in range(args.kernel_decomposition)] 
rdm_pairs = retrieve_dissimilarities(rdm, SI_DICT,abs_value=False)
kernel_pairs = [retrieve_dissimilarities(kernel, SI_DICT) for kernel in kernels]

all_pairs = set()
for kernel_pair_dict in kernel_pairs:
    all_pairs.update(kernel_pair_dict.keys())

top_kernels_per_pair = {pair: get_top_kernels_for_pair(pair, kernel_pairs,k = args.top_kernel_indexes) for pair in all_pairs}

top_kernels_values_per_pair = {pair: get_top_values_for_pair(pair, kernel_pairs,k = args.top_kernel_indexes) for pair in all_pairs}

SI_DICT_test1 = {pair: SI_DICT[(pair, 'test1')] for pair in set(pair_test[0] for pair_test in SI_DICT.keys()) if (pair, 'test1') in SI_DICT}

sorted_pairs_by_SI_test1 = sorted(SI_DICT_test1.keys(), key=lambda pair: SI_DICT_test1[pair])

#plot_overlapping_histograms(sorted_pairs_by_SI_test1, top_kernels_per_pair, kernels)

#plot_mean_std_scatter(sorted_pairs_by_SI_test1,top_kernels_per_pair,savedir = LOG_DIR)

#plot_mean_std_scatter(sorted_pairs_by_SI_test1,top_kernels_values_per_pair,type_plot='Values',savedir = LOG_DIR)

#plot_mean_std_vs_SI(sorted_pairs=sorted_pairs_by_SI_test1,top_kernels_per_pair=top_kernels_per_pair,SI_DICT=SI_DICT,savedir=LOG_DIR)

kernel_correlations_1 = []
kernel_correlations_dprime = []

kernel_correlations_2 = []
kernel_correlations_3 = []
kernel_correlations_4 = []

for i, kernel_pair in enumerate(kernel_pairs):
    if i<10:
       correlations = plot_si_vs_rdm(SI_DICT, kernel_pair, type=f'{i}_kernel',to_plot=True,save_dir=LOG_DIR)
       #correlations_drpime = plot_si_vs_rdm(SI_dict_dprime, kernel_pair, type=f'{i}_kernel_dprime',to_plot=True,save_dir=LOG_DIR)
       #kernel_correlations_dprime.append(correlations_drpime['test1'])
       kernel_correlations_1.append(correlations['test1'])
       kernel_correlations_2.append(correlations['test2'])
       kernel_correlations_3.append(correlations['test3'])
       kernel_correlations_4.append(correlations['test4'])
plot_kernel_correlations(kernel_correlations_1,title=f'{args.arch}_Correlation_Test_1',savedir=LOG_DIR)
#plot_kernel_correlations(kernel_correlations_dprime,title=f'{args.arch}_drpime_Correlation_Test_1 ',savedir=LOG_DIR)

plot_kernel_correlations(kernel_correlations_2,title=f'{args.arch}_Correlation Test 2',savedir=LOG_DIR)
plot_kernel_correlations(kernel_correlations_3,title=f'{args.arch}_Correlation Test 3',savedir=LOG_DIR)
plot_kernel_correlations(kernel_correlations_4,title=f'{args.arch}_Correlation Test 4',savedir=LOG_DIR)

torch.cuda.empty_cache()

print(f'TIME TOOK: {time.time()-start_time} ')

