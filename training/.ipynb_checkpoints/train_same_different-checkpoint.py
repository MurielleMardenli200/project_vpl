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
from tensorboardX import SummaryWriter
from stimupy.noises.whites import white
from database.coco import NSD,COCO,create_transform,create_transform_aperature,get_data_loaders,generate_test_transforms
import argparse
from paths import *
from models.model_arch import Model
from utilities.extract_activations import get_activation_list,save_act,get_layer_names
from utilities.plot_utils import (plot_si_vs_rdm,
                                  train_histogram,
                                  test_histogram,
                                  plot_kernel_correlations,
                                  fit_psychometric_curve,
                                  plot_overlapping_histograms,
                                  plot_mean_std_scatter,
                                  plot_mean_std_vs_SI,
                                  plot_all_pairs_losses,
                                  save_to_json)
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
                                   find_top_pairs_eigen_decomposition)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='VPL training')
parser.add_argument('--arch', type=str, default='alexnet')
parser.add_argument('--name', type=str, default='test_1')
parser.add_argument('--skip_connections_fc', default=False, action='store_true')
parser.add_argument('--skip_connections_conv2d', default=False, action='store_true')
parser.add_argument('--freeze_weights', default=True, action='store_true')
parser.add_argument('--log_interval', type=int, default=2000)
parser.add_argument('--trainset', type=str, default='COCO')
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--remove_fc', action='store_true', default=True)
parser.add_argument('--pretrained',action='store_true' , default=True)
parser.add_argument('--noise_type', type=str, default='None')
parser.add_argument('--intensity_range', nargs='+', type=int)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--task_prob', type=float, default=0.5)
parser.add_argument('--num_pairs', type=int, default=100)
parser.add_argument('--iterations', type=int, default=60)
parser.add_argument('--input_size_fc', type=int, default=9216)
parser.add_argument('--RDM_layer', type=str, default='avgpool')
parser.add_argument('--download_model',default=False, action='store_true')
parser.add_argument('--extreme_pairs', default=False, action='store_true')
parser.add_argument('--top_kernel_indexes', type=int, default=2)
parser.add_argument('--kernel_decomposition', type=int, default=999)
parser.add_argument('--arch_type', type=str, default='classifier')
parser.add_argument('--layers_to_remove', type=str, default='None')

args = parser.parse_args()

if args.arch_type == 'None':
    arch_type = None
else:
    arch_type = args.arch_type
layers_to_remove = []
if args.layers_to_remove == 'None':
    layers_to_remove = ['0','1','2','3','4','5','6']
else:
    layers_to_remove.append(args.layers_to_remove)

LOG_DIR = os.path.join(TENSORBOARD_DIR,f'{args.name}_{args.arch}_{args.lr}_{args.num_pairs}_{args.iterations}_{args.noise_type}_{args.batchsize}') 
os.makedirs(LOG_DIR, exist_ok=True)
model = Model(model_name=args.arch,pretrained=args.pretrained,remove_fc=args.remove_fc,skip_connections_fc=args.skip_connections_fc,skip_connections_conv2d=args.skip_connections_conv2d,device=device,layers_to_remove=layers_to_remove, arch_type = arch_type,input_size=args.input_size_fc,output_size=1,download_model=args.download_model).load().train().to(device)
print(model)
writer = SummaryWriter(LOG_DIR)

layer_names = get_layer_names(model)
save_act(model,layer_names=layer_names)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

print(device)

intensity_range = tuple(args.intensity_range)

configs_tasks= [
    {
        'noise_type': args.noise_type, 
        'intensity_range': (0,1), 
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
        'intensity_range': (0,0.5), 
        'type': 'circular_gaussian', 
        'radius': None, 
        'bg_size': (350, 350), 
        'position': (0, 0), 
        'zoom_factor': 1,
        'rotation': None,
        'horizontal_flip': None
    },
    {
    'noise_type': 'None', 
            'intensity_range': (0,1), 
            'type': 'circular_gaussian', 
            'radius': None, 
            'bg_size': (350, 350), 
            'position': (120, 120), 
            'zoom_factor': 1,
            'rotation': None,
            'horizontal_flip': None
        },
    {
    'noise_type': args.noise_type, 
            'intensity_range': (0,0.5), 
            'type': 'circular_gaussian', 
            'radius': None, 
            'bg_size': (350, 350), 
            'position': (0, 0), 
            'zoom_factor': 1,
            'rotation': None,
            'horizontal_flip': True
        },
    {
    'noise_type': args.noise_type, 
            'intensity_range': (0,0.5), 
            'type': 'circular_gaussian', 
            'radius': None, 
            'bg_size': (350, 350), 
            'position': (120, 120), 
            'zoom_factor': 1,
            'rotation': None,
            'vertical_flip': True
        },
]

train_task_config = configs_tasks[0]
IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN = generate_test_transforms(train_task_config)

dataset = NSD(data_dir=NSD_DIR,transforms=IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN)
images_dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False,)
model_activation = get_activation_list(model,images_dataloader,device=device)
my_act = torch.tensor(model_activation[args.RDM_layer]).reshape(1000,-1)

rdm = compute_rdm(my_act)
top_n = args.num_pairs
if args.extreme_pairs:
    top_100_pairs = find_extereme_pairs(rdm,top_n=top_n)
else: 
    top_100_pairs = find_top_pairs(rdm,top_n=top_n) 
pairs_train = top_100_pairs
#pairs_train = find_top_pairs_eigen_decomposition(top_100_pairs,rdm)

def test(model,pairs_trains,name_test,config_test):
    model.eval()
    criterion = nn.BCELoss()  
    IMAGENET_TRANSFORM_NOISE_APRATURE_TEST = generate_test_transforms(config_test)
    train_loader, _ = get_data_loaders(data_dir=COCO_DIR, transform=IMAGENET_TRANSFORM_NOISE_APRATURE_TEST, batch_size=args.batchsize, same_pair_probability=args.task_prob,same_not_rand=True, same_reference=False, ref_state=None, ref_other=None,idx_ref=pairs_trains[0],idx_other=pairs_trains[1])
    losses_dict = {}
    prob_list_test = []
    count = 0
    TRUE_val_list = []
    total_loss = 0

    for batch_idx, (img1_batch, img2_batch, label1_batch, label2_batch, same_pair) in enumerate(train_loader):
        if batch_idx == 1 or batch_idx == 2:
            img1_batch = img1_batch.to(device)
            img2_batch = img2_batch.to(device)
            label1_batch = label1_batch.to(device)
            label2_batch = label2_batch.to(device)

            output_1 = model(img1_batch)
            output_2 = model(img2_batch)

            _, porb = two_alt_forced_choise(output_1, output_2)

            prob_list_test.append(porb)

            TRUE_LABEL = torch.where(same_pair, torch.ones_like(same_pair), torch.zeros_like(same_pair))
            TRUE_LABEL = TRUE_LABEL.float().unsqueeze(1).to(device)

            TRUE_val_list.append(TRUE_LABEL)

            loss = criterion(porb, TRUE_LABEL)
            
            print(f'Loss at batch {batch_idx}: ', loss.item())

            total_loss = total_loss + loss.item()
            count += 1

            if batch_idx == 2:
                break

    if count > 0:
        average_loss = total_loss / count
        print('Average Loss: ', average_loss)
        return average_loss
    else:
        return None, None, None 

def register_test_loss(model, pairs_trains, configs_tasks, iteration, loss_test_dict, test_loss_per_iteration_dict):
    """
    Registers the test loss for each test configuration at a given iteration.

    :param model: The model being trained and tested.
    :param pairs_trains: The current pair of training data.
    :param configs_tasks: Dictionary of configurations for each test.
    :param iteration: The current iteration number in the training loop.
    :param loss_test_dict: Dictionary to hold the overall test loss for each test configuration.
    :param test_loss_per_iteration_dict: Dictionary to hold the test loss for each configuration at each iteration.
    """
    for test_name in ['test1', 'test2', 'test3', 'test4']:
        
        test_loss = test(model=model, pairs_trains=pairs_trains, name_test=test_name, config_test=configs_tasks[test_name])
        
        loss_test_dict[(tuple(pairs_trains), test_name)] = test_loss
        
        test_loss_per_iteration_key = (tuple(pairs_trains), test_name, iteration)
        test_loss_per_iteration_dict[test_loss_per_iteration_key] = test_loss
    return test_loss_per_iteration_dict

def train(pairs_train,config_train,iterations):
    loss_dict = {}
    prob_train_dict={}
    loss_test_dict = {}
    prob_test_dict = {}
    test_loss_per_iteration_dict = {}

    SI_dict = {}    
    for pairs_trains in pairs_train:
        
        IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN = generate_test_transforms(config_train)

        model = Model(model_name=args.arch,pretrained=args.pretrained,remove_fc=args.remove_fc,skip_connections_fc=args.skip_connections_fc,skip_connections_conv2d=args.skip_connections_conv2d,device=device,layers_to_remove=layers_to_remove, arch_type = arch_type,input_size=args.input_size_fc,output_size=1,download_model=args.download_model).load().train().to(device)

        criterion = nn.BCELoss()  

        freeze_layers_except_last(model,last_layer_name='fc')

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9) 
        
        train_loader, _ = get_data_loaders(data_dir=COCO_DIR, transform=IMAGENET_TRANSFORM_NOISE_APRATUR_TRAIN, batch_size=args.batchsize, same_pair_probability=args.task_prob,same_not_rand=True, same_reference=False, ref_state=None, ref_other=None,idx_ref=pairs_trains[0],idx_other=pairs_trains[1])

        lossess = []

        prob_list_train = []
        TRUE_train_list = []

        print(f'noise_type: {args.noise_type} max_intensity: {1}')
        iteration=0
        for batch_idx, (img1_batch, img2_batch, label1_batch, label2_batch, same_pair) in enumerate(train_loader):
            iteration=iteration+1
            while len(lossess) < iterations: 
                
                img1_batch = img1_batch.to(device)
                img2_batch = img2_batch.to(device)
                label1_batch = label1_batch.to(device)
                label2_batch = label2_batch.to(device)

                output_1 = model(img1_batch)
                output_2 = model(img2_batch)

                delta_h, porb = two_alt_forced_choise(output_1, output_2)

                prob_list_train.append(porb)

                TRUE_LABEL = torch.where(same_pair, torch.ones_like(same_pair), torch.zeros_like(same_pair))
                TRUE_LABEL = TRUE_LABEL.float().unsqueeze(1).to(device)

                TRUE_train_list.append(TRUE_LABEL.cpu())

                loss = criterion(porb, TRUE_LABEL)

                lossess.append(loss.item())

                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()

                print('loss: ', loss)
                #test_loss_per_iteration_dict = register_test_loss(model, pairs_trains, configs_tasks, iteration, loss_test_dict, test_loss_per_iteration_dict)
                
                #train_histogram(noise_type=args.noise_type,intensity_range=intensity_range,porb=porb,TRUE_LABEL=TRUE_LABEL,step=i,pairs=pairs_trains)

        loss_test_dict[tuple(pairs_trains), 'test1']=test(model=model,pairs_trains=pairs_trains,name_test='test1',config_test=configs_tasks[1])
        loss_test_dict[tuple(pairs_trains), 'test2']=test(model=model,pairs_trains=pairs_trains,name_test='test2',config_test=configs_tasks[2])
        loss_test_dict[tuple(pairs_trains), 'test3']=test(model=model,pairs_trains=pairs_trains,name_test='test3',config_test=configs_tasks[3])
        loss_test_dict[tuple(pairs_trains), 'test4']=test(model=model,pairs_trains=pairs_trains,name_test='test4',config_test=configs_tasks[4])
        loss_dict[(pairs_trains, (0,1))] = lossess
   
        SI_dict[tuple(pairs_trains), 'test1'] = calculate_specificity_loss(test=loss_test_dict[tuple(pairs_trains), 'test1'],train=loss_dict[(pairs_trains, (0,1))])
        SI_dict[tuple(pairs_trains), 'test2'] = calculate_specificity_loss(test=loss_test_dict[tuple(pairs_trains), 'test2'],train=loss_dict[(pairs_trains, (0,1))])
        SI_dict[tuple(pairs_trains), 'test3'] = calculate_specificity_loss(test=loss_test_dict[tuple(pairs_trains), 'test3'],train=loss_dict[(pairs_trains, (0,1))])
        SI_dict[tuple(pairs_trains), 'test4'] = calculate_specificity_loss(test=loss_test_dict[tuple(pairs_trains), 'test4'],train=loss_dict[(pairs_trains, (0,1))])

    print('DONE')
    return SI_dict, loss_dict, test_loss_per_iteration_dict

iterations = args.iterations
SI_DICT,loss_dict, test_loss_per_iteration_dict  = train(pairs_train=pairs_train,config_train=configs_tasks[0],iterations=iterations)
save_to_json(loss_dict, SI_DICT, rdm)

plot_all_pairs_losses(loss_dict,color_split=True)
kernels = [rdm_decompostion(kernel_num=i, rdm=rdm) for i in range(args.kernel_decomposition)] #10 
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

plot_mean_std_scatter(sorted_pairs_by_SI_test1,top_kernels_per_pair)

plot_mean_std_scatter(sorted_pairs_by_SI_test1,top_kernels_values_per_pair,type_plot='Values')

plot_mean_std_vs_SI(sorted_pairs=sorted_pairs_by_SI_test1,top_kernels_per_pair=top_kernels_per_pair,SI_DICT=SI_DICT)

kernel_correlations_1 = []
kernel_correlations_2 = []
kernel_correlations_3 = []
kernel_correlations_4 = []


for i, kernel_pair in enumerate(kernel_pairs):
    if i<10:
        correlations = plot_si_vs_rdm(SI_DICT, kernel_pair, type=f'{i}_kernel',to_plot=True)
        kernel_correlations_1.append(correlations['test1'])
        kernel_correlations_2.append(correlations['test2'])
        kernel_correlations_3.append(correlations['test3'])
        kernel_correlations_4.append(correlations['test4'])
plot_kernel_correlations(kernel_correlations_1,title=f'{args.arch}_Correlation Test 1')
plot_kernel_correlations(kernel_correlations_2,title=f'{args.arch}_Correlation Test 2')
plot_kernel_correlations(kernel_correlations_3,title=f'{args.arch}_Correlation Test 3')
plot_kernel_correlations(kernel_correlations_4,title=f'{args.arch}_Correlation Test 4')

torch.cuda.empty_cache()

