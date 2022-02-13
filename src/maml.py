import argparse

import numpy as np
import os
import tqdm
import torch
from torch import nn, optim

from src.zoo.maml_utils import inner_adapt_maml, setup
from src.utils2 import Profiler
#from src.config import maml_omniglot, maml_mini

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--root', type=str)
parser.add_argument('--n-ways', type=int)
parser.add_argument('--k-shots', type=int)
parser.add_argument('--q-shots', type=int)
parser.add_argument('--model-path', type=str)
parser.add_argument('--inner-adapt-steps-train', type=int)
parser.add_argument('--inner-adapt-steps-test', type=int)
parser.add_argument('--inner-lr', type=float)
parser.add_argument('--meta-lr', type=float)
parser.add_argument('--meta-batch-size', type=int)
parser.add_argument('--iterations', type=int)
parser.add_argument('--order', type=str)
parser.add_argument('--device', type=str)
parser.add_argument('--experiment', type=str)

args = parser.parse_args()
if args.order == 'True': args.order = True
elif args.order == 'False': args.order = False

if args.download == 'True': args.download = True
elif args.download == 'False': args.download = False


# Generating Tasks, initializing learners, loss, meta - optimizer
train_tasks, valid_tasks, test_tasks, learner = setup(
    args.dataset, args.root, args.n_ways, args.k_shots, args.q_shots, args.order, args.inner_lr, args.device, download=args.download)
opt = optim.Adam(learner.parameters(), args.meta_lr)
loss = nn.CrossEntropyLoss(reduction='mean')
if args.order == False:
    profiler = Profiler('MAML_{}_{}-way_{}-shot_{}-queries'.format(args.dataset, args.n_ways, args.k_shots, args.q_shots))
    prof_test = Profiler('MAML_test_{}_{}-way_{}-shot_{}-queries'.format(args.dataset, args.n_ways, args.k_shots, args.q_shots))
elif args.order == True:
    profiler = Profiler('FO-MAML_{}_{}-way_{}-shot_{}-queries'.format(args.dataset, args.n_ways, args.k_shots, args.q_shots))
    prof_test = Profiler('FO-MAML_test_{}_{}-way_{}-shot_{}-queries'.format(args.dataset, args.n_ways, args.k_shots, args.q_shots))

    
## Training ##
for iter in tqdm.tqdm(range(args.iterations)):
    opt.zero_grad()
    meta_train_loss = []
    meta_valid_loss = []
    meta_train_acc = []
    meta_valid_acc = []
    batch_losses = []


    for batch in range(args.meta_batch_size):
        ttask = train_tasks.sample()
        model = learner.clone()
        evaluation_loss, evaluation_accuracy, _, _ = inner_adapt_maml(
            ttask, loss, model, args.n_ways, args.k_shots, args.q_shots, args.inner_adapt_steps_train, args.device)
        evaluation_loss.backward()
        tmp = [(iter*args.meta_batch_size)+batch, evaluation_accuracy.item()]
        batch_losses.append(tmp)

    vtask = valid_tasks.sample()
    model = learner.clone()
    validation_loss, validation_accuracy, _, _ = inner_adapt_maml(
        vtask, loss, model, args.n_ways, args.k_shots, args.q_shots, args.inner_adapt_steps_train, args.device)
    
    
    for p in learner.parameters():
        p.grad.data.mul_(1.0 / args.meta_batch_size)
    opt.step()

    profiler.log_csv(batch_losses, 'train')
    # Checkpointing the learner
    if iter % 500 == 0:
        learner = learner.to('cpu')
        profiler.log_model(learner, opt, iter)
        learner = learner.to(args.device)
    else:
        continue


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

