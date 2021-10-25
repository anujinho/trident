import os
import argparse
import json
from torch._C import device

#import numpy as np
import tqdm
import torch
from torch import nn

from src.utils2 import Profiler
from src.zoo.delpo_utils import inner_adapt_delpo, setup

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
parser.add_argument('--n-ways', type=int)
parser.add_argument('--k-shots', type=int)
parser.add_argument('--q-shots', type=int)
parser.add_argument('--inner-adapt-steps-train', type=int)
parser.add_argument('--inner-adapt-steps-test', type=int)
parser.add_argument('--inner-lr', type=float)
parser.add_argument('--meta-lr', type=float)
parser.add_argument('--meta-batch-size', type=int)
parser.add_argument('--iterations', type=int)
parser.add_argument('--wt-ce', type=float)
parser.add_argument('--klwt', type=str)
parser.add_argument('--rec-wt', type=float)
parser.add_argument('--beta-l', type=float)
parser.add_argument('--beta-s', type=float)
parser.add_argument('--experiment', type=str)
parser.add_argument('--order', type=str)
parser.add_argument('--device', type=str)
parser.add_argument('--download', type=str)
parser.add_argument('--repar', type=str, default=True)
parser.add_argument('--resume', type=str)
parser.add_argument('--iter-resume', type=int)

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

if args.download == 'True':
    args.download = True
elif args.download == 'False':
    args.download = False

if args.klwt == 'True':
    args.klwt = True
elif args.klwt == 'False':
    args.klwt = False

# wandb.config.update(args)

# Generating Tasks, initializing learners, loss, meta - optimizer and profilers
train_tasks, valid_tasks, test_tasks, _ = setup(
    args.dataset, args.root, args.n_ways, args.k_shots, args.q_shots, args.order, args.inner_lr, args.device, download=args.download, repar=True)
reconst_loss = nn.MSELoss(reduction='none')
if args.order == False:
    profiler = Profiler('DELPO_test_{}_{}-way_{}-shot_{}-queries'.format(args.dataset,
                        args.n_ways, args.k_shots, args.q_shots), args.experiment)

elif args.order == True:
    profiler = Profiler('FO-DELPO_{}_{}-way_{}-shot_{}-queries'.format(
        args.dataset, args.n_ways, args.k_shots, args.q_shots), args.experiment)


## Testing ##

for model_name in os.listdir(args.model_path):
    learner = torch.load('{}/{}'.format(args.model_path, model_name))
    learner = learner.to(args.device)
    print('Testing on held out classes')

    for i, tetask in enumerate(test_tasks):
        # wandb.define_metric("accuracies", summary="max")
        # wandb.define_metric("accuracies", summary="mean")

        model = learner.clone()
        #tetask = test_tasks.sample()
        evaluation_loss, evaluation_accuracy = inner_adapt_delpo(
            tetask, reconst_loss, model, args.n_ways, args.k_shots, args.q_shots, args.inner_adapt_steps_test, args.device, False, args)

        # Logging per test-task losses and accuracies
        tmp = [i, evaluation_accuracy.item()]
        tmp = tmp + [a.item() for a in evaluation_loss.values()]
        profiler.log_csv(tmp, 'test')
        # wandb.log(dict({f"test/{key}": loss.item() for _, (key, loss) in enumerate(evaluation_loss.items())},
        #             **{'test/accuracies': evaluation_accuracy.item(), 'test/task': i}))
