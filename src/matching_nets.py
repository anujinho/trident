import argparse

#import learn2learn as l2l
import numpy as np
import torch
#import torch.nn.functional as F
import tqdm
from torch import nn, optim

from src.zoo.matching_nets_utils import inner_adapt_matching, setup
from src.utils import Profiler

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--root', type=str)
parser.add_argument('--n-ways', type=int)
parser.add_argument('--k-shots', type=int)
parser.add_argument('--q-shots', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--meta-batch-size', type=int)
parser.add_argument('--iterations', type=int)
parser.add_argument('--layers', type=int)
parser.add_argument('--unrolling-steps', type=int)
parser.add_argument('--device', type=str)

args = parser.parse_args()
EPSILON = 1e-8

# Generating Tasks, initializing learners, loss, meta - optimizer
train_tasks, valid_tasks, test_tasks, learner = setup(
    args.dataset, args.root, args.n_ways, args.k_shots, args.q_shots, args.layers, args.unrolling_steps, args.device)
opt = optim.Adam(learner.parameters(), args.lr)
loss = nn.NLLLoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    opt, step_size=20, gamma=0.5)
profiler = Profiler('MatchingNets_{}_{}-shot_{}-way_{}-queries'.format(
    args.dataset, args.n_ways, args.k_shots, args.q_shots))

## Training ##
for iter in tqdm.tqdm(range(args.iterations)):

    meta_train_loss = []
    meta_valid_loss = []
    meta_train_acc = []
    meta_valid_acc = []
    learner.train()

    for batch in range(args.meta_batch_size):
        opt.zero_grad()
        ttask = train_tasks.sample()
        evaluation_loss, evaluation_accuracy = inner_adapt_matching(
            ttask, loss, learner, args.n_ways, args.k_shots, args.q_shots, EPSILON, args.device)
        meta_train_loss.append(evaluation_loss.item())
        meta_train_acc.append(evaluation_accuracy.item())
        evaluation_loss.backward()
        torch.nn.clip_grad_norm_(learner.parameters(), 1)
        opt.step()
    lr_scheduler.step()

    learner.eval()
    for i, vtask in enumerate(valid_tasks):
        validation_loss, validation_accuracy = inner_adapt_matching(
            vtask, loss, learner, args.n_ways, args.k_shots, args.q_shots, args.device)
        meta_valid_loss.append(validation_loss.item())
        meta_valid_acc.append(validation_accuracy.item())

    profiler.log(row = [np.array(meta_train_acc).mean(), np.array(meta_train_acc).std(), np.array(meta_train_loss).mean(), np.array(meta_train_loss).std(), np.array(
        meta_valid_acc).mean(), np.array(
        meta_valid_acc).std(), np.array(
        meta_valid_loss).mean(), np.array(
        meta_valid_loss).std()])

    if (iter % 10 == 0):
        print('Meta Train Accuracy: {:.4f} +- {:.4f}'.format(
            np.array(meta_train_acc).mean(), np.array(meta_train_acc).std()))
        print('Meta Valid Accuracy: {:.4f} +- {:.4f}'.format(
            np.array(meta_valid_acc).mean(), np.array(meta_valid_acc).std()))


## Testing ##
prof_test = Profiler('MatchingNets_test_{}_{}-shot_{}-way_{}-queries'.format(
    args.dataset, args.n_ways, args.k_shots, args.q_shots))
print('Testing on held out classes')
for i, tetask in enumerate(test_tasks):
    meta_test_acc = []
    meta_test_loss = []
    evaluation_loss, evaluation_accuracy = inner_adapt_matching(
        tetask, loss, learner, args.n_ways, args.k_shots, args.q_shots, args.device)
    meta_test_loss.append(evaluation_loss.item())
    meta_test_acc.append(evaluation_accuracy.item())
    prof_test.log(row = [np.array(meta_test_acc).mean(), np.np.array(meta_test_acc).std(
    ), np.array(meta_test_loss).mean(), np.np.array(meta_test_loss).std()])
    print('Meta Test Accuracy', np.array(meta_test_acc).mean(),
          '+-', np.np.array(meta_test_acc).std())
