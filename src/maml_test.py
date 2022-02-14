import os
import argparse
import json
import numpy as np
from torch._C import device

import tqdm
import torch
from torch import nn

from src.utils2 import Profiler
from src.zoo.maml_utils import inner_adapt_maml, setup

#import wandb

#wandb.init(project="meta", entity='anujinho', config={})

##############
# Parameters #
##############

parser = argparse.ArgumentParser()
parser.add_argument('--cnfg', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--root', type=str)
parser.add_argument('--model-path', type=str)
parser.add_argument('--backbone', type=list)
parser.add_argument('--n-ways', type=int)
parser.add_argument('--k-shots', type=int)
parser.add_argument('--q-shots', type=int)
parser.add_argument('--inner-adapt-steps-test', type=int)
parser.add_argument('--inner-lr', type=float)
parser.add_argument('--meta-lr', type=float)
parser.add_argument('--meta-batch-size', type=int)
parser.add_argument('--iterations', type=int)
parser.add_argument('--order', type=str)
parser.add_argument('--device', type=str)
parser.add_argument('--experiment', type=str)
parser.add_argument('--times', type=int)
parser.add_argument('--extra', type=str)


args = parser.parse_args()
with open(args.cnfg) as f:
    parser = argparse.ArgumentParser()
    argparse_dict = vars(args)
    argparse_dict.update(json.load(f))

    args = argparse.Namespace()
    args.__dict__.update(argparse_dict)


# TODO: fix this bool/str shit

if args.order == 'True':
    args.order = True
elif args.order == 'False':
    args.order = False

if args.backbone[0] == 'True':
    args.backbone[0] = True
elif args.backbone[0] == 'False':
    args.backbone[0] = False

# Generating Tasks, initializing learners, loss, meta - optimizer
_, _, test_tasks, learner = setup(
    args.dataset, args.root, args.n_ways, args.k_shots, args.q_shots, args.order, args.inner_lr, args.device, download=False)
loss = nn.CrossEntropyLoss(reduction='mean')
if args.order == False:
    profiler = Profiler('MAML_test_{}_{}-way_{}-shot_{}-queries'.format(args.dataset, args.n_ways, args.k_shots, args.q_shots), args.experiment, args)
elif args.order == True:
    profiler = Profiler('FO-MAML_test_{}_{}-way_{}-shot_{}-queries'.format(args.dataset, args.n_ways, args.k_shots, args.q_shots))


## Testing ##

for model_name in os.listdir(args.model_path):
    learner = torch.load('{}/{}'.format(args.model_path, model_name))
    learner = learner.to(args.device)
    print('Testing on held out classes')
    for i, tetask in enumerate(test_tasks):
        
        model = learner.clone()
        #tetask = test_tasks.sample()
        evaluation_loss, evaluation_accuracy, logits, labels = inner_adapt_maml(
            tetask, loss, model, args.n_ways, args.k_shots, args.q_shots, args.inner_adapt_steps_test, args.device)
        
        # Logging test-task logits and ground-truth labels
        tmp = np.array(torch.cat([torch.full((args.n_ways*args.q_shots, 1), i), logits, labels.unsqueeze(dim=1)], axis=1))
        profiler.log_csv(tmp, 'preds')
        
        # Logging per test-task losses and accuracies
        tmp = [i, evaluation_accuracy.item()]
        tmp = tmp + [model_name]
        profiler.log_csv(tmp, 'test')
