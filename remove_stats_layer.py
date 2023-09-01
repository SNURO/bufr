from __future__ import division, print_function, absolute_import
import argparse
import yaml
import time
from nets import MNISTCNNBase, ResNet18, learner_distances, add_stats_layer_to_resnet_named_modules, \
    add_stats_layers_to_cnn_classifier, add_stats_layers_to_cnn_everywhere
import nets_wilds
from lib.utils import *
from lib.stats_layers import *
from lib.data_utils import get_static_emnist_dataloaders, get_static_emnist_dataloaders_fewshot, \
    get_cifar10c_dataloaders, get_cifar100c_dataloaders, per_hospital_wilds_dataloader, \
    per_hospital_wilds_dataloader_fewshot
from data.digits import *
import torch.nn as nn

ckpt_dir = "/home2/wonjae.roh/nprc/bufr/ckpts/cifar10/adapted-learner-30_gaussian_noise_sgd_0.1_BUFR_fc_0.01_123.pth.tar"


learner = ResNet18(n_classes=10)
modules_to_track = ['linear']
module_features_out = [10]
module_features_in = [512]

# Add stats layers to model (*before* loading weights and units stats)--- 
# model ckpt를 load하려면 일단 이 과정을 거쳐야 함, 그 후에 지우고 다시 저장 하더라도.


stats_layers = ["soft_bins"]

if len(stats_layers) > 0:
    for stats_layer in stats_layers:
        add_stats_layer_to_resnet_named_modules(learner, modules_to_track, module_features_out,
                                                module_features_in, stats_layer_type=stats_layer,
                                                surprise_score="PSI",
                                                tau=0.01)
        for learner_stats_layer in learner.stats_layers:
            learner_stats_layer.calc_surprise = True
        learner_stats_layers = learner.stats_layers

## now learner 은 ready to load ckpt

dev=torch.device('cpu')
learner = learner.to(dev)
_, learner = load_ckpt('adapted-learner', learner, ckpt_dir, dev)


#import ipdb; ipdb.set_trace()

# remove stats layer from model
sequantial_list = list(learner.linear)
del sequantial_list[2]
del sequantial_list[0]
learner.linear=sequantial_list[0]
learner_stats_layers =[]

## 이제 다시 아무일도 없었던 것처럼 저장해놓자 -> 처음부터 과정을 반복할 수 있도록
save_dir = "/home2/wonjae.roh/nprc/bufr/ckpts/cifar10"
save_ckpt(save_dir, 'adapted-learner', learner, None)