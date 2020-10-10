# Copyright 2020 Jaime Tierney, Adam Luchies, and Brett Byram

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the license at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and 
# limitations under the License.

#!/usr/bin/env python

import torch
import os
import numpy as np
import warnings
import time
import argparse
import json
import sys
from pprint import pprint

from utils import read_model_params, save_model_params, ensure_dir, add_suffix_to_path
from dataloader import ApertureDataset
from model import FullyConnectedNet
from logger import Logger
from trainer import Trainer


if __name__ == '__main__':

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_params_path', help='Option to load model params from a file. Values in this file take precedence.')
    parser.add_argument('-k', default=4, help='Integer value. DFT frequency to analyze.', type=int)
    parser.add_argument('-s', '--save_dir', default=None, help='Directory to save the model.')
    parser.add_argument('-b', '--batch_size', default=100, help='Option to specify batch size.', type=int, )
    parser.add_argument('--data_noise_gaussian', help='Option to enable gaussian noise in channel data during training.', default=0, type=int)
    parser.add_argument('--dropout_input', help='Specify dropout probability for hidden nodes', default=0, type=int)
    parser.add_argument('-p', '--patience', type=int, default=20, help='Option to patience.')
    parser.add_argument('-c', '--cuda', help='Option to use GPU.', default=0, type=int)
    parser.add_argument('--save_initial', help='Option to save initial checkpoint of the model.', default=0, type=int)
    parser.add_argument('--input_dim', help='Specify input dimension for the network', default=130, type=int)
    parser.add_argument('--output_dim', help='Specify output dimension for the network', default=130, type=int)
    parser.add_argument('--layer_width', help='Specify layer_width for the network', default=260, type=int)
    parser.add_argument('--dropout', help='Specify dropout probability for hidden nodes', default=0, type=float)
    parser.add_argument('--weight_decay', help='Specify weight decay for hidden nodes', default=0, type=float)
    parser.add_argument('--num_hidden', help='Specify number of hidden layers', default=1, type=int)
    parser.add_argument('--data_is_target', help='Specify if targets are input data (autoencoder option).', default=0, type=int)
    parser.add_argument('--num_samples_train', help='Specify number of samples to use during training.', default=1000, type=int)
    parser.add_argument('--num_samples_val', help='Specify number of samples to use during validation.', default=10000, type=int)
    parser.add_argument('--starting_weights', help='Specify path/file for model starting weights.', default=None)
    parser.add_argument('--loss_function', help='Specify loss function.', default='MSELoss')
    args = parser.parse_args()

    # load model params if it is specified
    if args.model_params_path:
        model_params = read_model_params(args.model_params_path)
    else:
        model_params = {}

    # merge model_params and input args, giving preference to model_params
    model_params = {**vars(args), **model_params}

    # cuda flag
    print('torch.cuda.is_available(): ' + str(torch.cuda.is_available()))
    if model_params['cuda'] and torch.cuda.is_available():
        print('Using ' + str(torch.cuda.get_device_name(0)))
    else:
        print('Not using CUDA')
        model_params['cuda']=False

    # set device based on cuda flag
    device = torch.device("cuda:0" if model_params['cuda'] else "cpu")

    # load training data specification file
    with open(model_params['training_data_file'], 'r') as f:
        data_json = json.load(f)


    # Load primary training data
    dat_list = []

    for item in data_json['train']:
        fname = item['file']
        N = int( item['N'])
        dat_list.append( ApertureDataset(fname, N, model_params['k'], model_params['data_is_target']) )

    # print datasets
    print('\nTrain Data:')
    for dat in dat_list:
        print(dat)
    print('\n')

    dat_train = torch.utils.data.ConcatDataset(dat_list)


    # Load eval training data
    dat_list = []

    for item in data_json['train_eval']:
        fname = item['file']
        N = int( item['N'])
        dat_list.append( ApertureDataset(fname, N, model_params['k'], model_params['data_is_target']) )

    # print datasets
    print('\nTrain Eval Data:')
    for dat in dat_list:
        print(dat)
    print('\n')

    dat_eval = torch.utils.data.ConcatDataset(dat_list)


    # Load val data
    dat_list = []

    for item in data_json['val']:
        fname = item['file']
        N = int( item['N'])
        dat_list.append( ApertureDataset(fname, N, model_params['k'], model_params['data_is_target']) )

    # print datasets
    print('\nValidation Data:')
    for dat in dat_list:
        print(dat)
    print('\n')

    dat_val = torch.utils.data.ConcatDataset(dat_list)



    # setup data loaders
    last_batch_size = (len(dat_train) % model_params['batch_size'])
    last_batch_size = model_params['batch_size'] if (last_batch_size == 0) else last_batch_size
    print(f"\nLast batch size for train data: {last_batch_size}\n")
    drop_last = True if ( last_batch_size == 1) else False
    print(f"Drop last batch: {drop_last}")
    loader_train = torch.utils.data.DataLoader(dat_train, batch_size=model_params['batch_size'],
                                                shuffle=True, num_workers=1, drop_last=drop_last)

    drop_last=False
    loader_train_eval = torch.utils.data.DataLoader(dat_eval, batch_size=len(dat_eval), shuffle=False,
                                                        num_workers=1, drop_last=drop_last)

    drop_last=False
    loader_val = torch.utils.data.DataLoader(dat_val, batch_size=len(dat_val), shuffle=False,
                                                        num_workers=1, drop_last=drop_last)



    # create model
    model = FullyConnectedNet(input_dim=model_params['input_dim'],
                                output_dim=model_params['output_dim'],
                                layer_width=model_params['layer_width'],
                                dropout=model_params['dropout'],
                                dropout_input=model_params['dropout_input'],
                                num_hidden=model_params['num_hidden'],
                                starting_weights=model_params['starting_weights'],
                                batch_norm_enable=model_params['batch_norm_enable'])

    # save initial weights
    if model_params['save_initial'] and model_params['save_dir']:
        suffix = '_initial'
        path = add_suffix_to_path(model_parmas['save_dir'], suffix)
        print('Saving model weights in : ' + path)
        ensure_dir(path)
        torch.save(model.state_dict(), os.path.join(path, 'model.dat'))
        save_model_params(os.path.join(path, 'model_params.txt'), model_params)

    # loss
    if model_params['loss_function'] == 'L1Loss':
        loss = torch.nn.L1Loss()
    elif model_params['loss_function'] == 'MSELoss':
        loss = torch.nn.MSELoss()
    elif model_params['loss_function'] == 'SmoothL1Loss':
        loss = torch.nn.SmoothL1Loss()

    # optimizer
    #optimizer = torch.optim.SGD(model.parameters(),
    #                            lr=model_params['lr'],
    #                            momentum=model_params['momentum'])

    optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=model_params['lr'],
                                    betas=(model_params['beta1'], model_params['beta2']),
                                    weight_decay=model_params['weight_decay'])

    # scheduler
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                mode='min',
    #                                factor=0.1,
    #                                patience=model_params['lr_patience'],
    #                                min_lr=10**-7,
    #                                verbose=True)
    scheduler = None

    # logger
    logger = Logger()

    # send things to gpu if enabled
    loss = loss.to(device)
    model = model.to(device)

    # update model params
    model_params['num_samples_train'] = len(dat_train)
    model_params['num_samples_train_eval'] = len(dat_eval)
    model_params['num_samples_val'] = len(dat_val)
    model_params_path = os.path.join(model_params['save_dir'], 'model_params.txt')
    model_params['model_params_path'] = model_params_path
    if model_params['save_dir']:
        ensure_dir(model_params['save_dir'])
        save_model_params(model_params_path, model_params)

    # display input arguments
    print('\n')
    pprint(model_params)
    print('\n')

    # trainer
    trainer = Trainer(model=model,
                        loss=loss,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        patience=model_params['patience'],
                        loader_train=loader_train,
                        loader_train_eval=loader_train_eval,
                        loader_val=loader_val,
                        cuda=model_params['cuda'],
                        logger=logger,
                        data_noise_gaussian=model_params['data_noise_gaussian'],
                        save_dir=model_params['save_dir'])

    # run training
    trainer.train()
