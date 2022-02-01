import argparse
import json
import os, glob

import numpy as np
import tqdm
import torch
from torch import nn, optim

from src.utils2 import Profiler
from src.zoo.delpo_epi_utils import inner_adapt_delpo, setup

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
parser.add_argument('--meta-lr', type=float)
parser.add_argument('--meta-batch-size', type=int)
parser.add_argument('--iterations', type=int)
parser.add_argument('--wt-ce', type=float)
parser.add_argument('--klwt', type=str)
parser.add_argument('--rec-wt', type=float)
parser.add_argument('--beta-l', type=float)
parser.add_argument('--beta-s', type=float)
parser.add_argument('--task-adapt', type=str)
parser.add_argument('--task-adapt-fn', type=str)
parser.add_argument('--alpha', type=float)
parser.add_argument('--experiment', type=str)
parser.add_argument('--device', type=str)
parser.add_argument('--download', type=str)
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

if args.download == 'True':
    args.download = True
elif args.download == 'False':
    args.download = False

if args.klwt == 'True':
    args.klwt = True
elif args.klwt == 'False':
    args.klwt = False

if args.task_adapt == 'True':
    args.task_adapt = True
elif args.task_adapt == 'False':
    args.task_adapt = False

if args.backbone[0] == 'True':
    args.backbone[0] = True
elif args.backbone[0] == 'False':
    args.backbone[0] = False

# wandb.config.update(args)

# Generating Tasks, initializing learners, loss, meta - optimizer and profilers
train_tasks, valid_tasks, _, learner = setup(
    args.dataset, args.root, args.n_ways, args.k_shots, args.q_shots, args.device, download=args.download, task_adapt=args.task_adapt, task_adapt_fn=args.task_adapt_fn, args=args)
opt = optim.Adam(learner.parameters(), args.meta_lr)
reconst_loss = nn.MSELoss(reduction='none')

if args.resume == 'Yes':
    learner = learner.to('cpu')
    dict_model = torch.load('{}/model_{}.pt'.format(args.model_path, args.iter_resume)).state_dict()
    learner.load_state_dict(dict_model)
    learner = learner.to(args.device)
    opt.load_state_dict(torch.load('{}/opt_{}.pt'.format(args.model_path, args.iter_resume)))
    learner.train()
    start = args.iter_resume + 1

else:
    start = 0

profiler = Profiler('DELPO_Epi_{}_{}-way_{}-shot_{}-queries'.format(args.dataset,
                    args.n_ways, args.k_shots, args.q_shots), args.experiment, args)
folder = 'DELPO_Epi_{}_{}-way_{}-shot_{}-queries'.format(args.dataset,
                    args.n_ways, args.k_shots, args.q_shots)

## Training ##
val_acc_prev = 0
for iter in tqdm.tqdm(range(start, args.iterations)):
    opt.zero_grad()
    batch_losses = []
    val_losses = []

    for batch in range(args.meta_batch_size):
        ttask = train_tasks.sample()        
        evaluation_loss, evaluation_accuracy = inner_adapt_delpo(
            ttask, reconst_loss, model, args.n_ways, args.k_shots, args.q_shots, args.device, False, args)
        
        # Logging per train-task losses and accuracies
        tmp = [(iter*args.meta_batch_size)+batch, evaluation_accuracy.item()]
        tmp = tmp + [a.item() for a in evaluation_loss.values()]
        batch_losses.append(tmp)

        # Backprop + GD Step
        evaluation_loss['elbo'].backward()
        opt.step()

    for batch in range(500):
        vtask = valid_tasks.sample()
        model = learner.clone()

        validation_loss, validation_accuracy = inner_adapt_delpo(
            vtask, reconst_loss, model, args.n_ways, args.k_shots, args.q_shots, args.inner_adapt_steps_train, args.device, False, args)

        # Logging per validation-task losses and accuracies
        tmp = [(iter*500)+batch, validation_accuracy.item()]
        tmp = tmp + [a.item() for a in validation_loss.values()]
        val_losses.append(tmp)

    # Saving the Logs
    profiler.log_csv(batch_losses, 'train')
    profiler.log_csv(val_losses, 'valid')

    # Checkpointing the learner
    if (iter == 0) or (np.array(val_losses)[:, 1].mean() >= val_acc_prev):
        learner = learner.to('cpu')
        for filename in glob.glob("/users/anujsingh/files/learning_to_meta-learn/logs/{}/{}/model*".format(folder, args.experiment)):
            os.remove(filename)
        for filename in glob.glob("/users/anujsingh/files/learning_to_meta-learn/logs/{}/{}/opt*".format(folder, args.experiment)):
            os.remove(filename) 

        profiler.log_model(learner, opt, iter)
        learner = learner.to(args.device)
    else:
        continue
    val_acc_prev = np.array(val_losses)[:, 1].mean()

profiler.log_model(learner, opt, 'last')
profiler.log_model(learner, opt, iter)