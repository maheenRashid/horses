#!/usr/bin/env python2.7
# coding: utf-8
"""
    This file generate prototxt file for LEVEL-2 and LEVEL-3
"""

import sys

def generate(device_num, 
            dataset_name, 
            net_type, 
            n_epochs,
            base_lr,
            train_batch_size,
            weights,
            restore,
            period_num,
            lr_mults,
            lr_prefix,
            new_layers,
            display, 
            snapshot_interval_epoch):
    """
        Generate template
        see train_template.py.template
    """
    templateFile = 'train_script.py.template'
    with open(templateFile, 'r') as fd:
        template = fd.read()
        outputFile = dataset_name+'_'+net_type+'_'+lr_prefix+'.py'
        with open(outputFile, 'w') as fd:
            fd.write(template.format(device_num=str(device_num), 
                                    dataset_name=dataset_name, 
                                    net_type=net_type, 
                                    n_epochs=str(n_epochs),
                                    base_lr=str(base_lr),
                                    train_batch_size=str(train_batch_size),
                                    weights=weights,
                                    restore=restore,
                                    period_num=str(period_num),
                                    lr_mults=lr_mults,
                                    lr_prefix=lr_prefix,
                                    new_layers=new_layers,
                                    display=str(display),
                                    snapshot_interval_epoch=str(snapshot_interval_epoch)))


if __name__ == '__main__':
    small_net_lr_mults = "{'Conv1':1, 'Conv2':1, 'Conv3':1, 'Conv4':1, 'fc1':1, 'fc2':1}"
    small_net_new_layers = "{'fc2':'h_fc2'}"
    
    caffenet_lr_mults = "{'conv1':1, 'conv2':1, 'conv3':1, 'conv4':1, 'conv5':1, 'fc6':1, 'fc7':1, 'fc8':1}"
#    caffenet_new_layers = "{'fc6':'human_fc6', 'fc7':'human_fc7', 'fc8':'human_fc8'}"
    caffenet_new_layers = "{'fc8':'human_fc8'}"
    
    caffenet_init_weights = '/home/laoreja/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    vanilla_init_weights = '/home/laoreja/data/VanillaCNN/ZOO/vanillaCNN.caffemodel'
    
    generate(
        device_num=1, 
        dataset_name='aflw_40_vanilla',
        # 'aflw_5_points_224_vanilla', 
        net_type='selected_drop_smooth_vanilla',
        # 'normalized_caffenet', 
        n_epochs=20,
        base_lr=0.00001,
        train_batch_size=64,
        weights=vanilla_init_weights,
        # caffenet_init_weights,
        restore='',
        period_num=0,
        lr_mults=small_net_lr_mults,
        # caffenet_lr_mults,
        lr_prefix='learn_all_layers_bs_64',
        new_layers=small_net_new_layers,
        # caffenet_new_layers,
        display=200,
        snapshot_interval_epoch=1)

    # Done
