import argparse
import copy

#import learn2learn as l2l
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from src.zoo.lpo_utils import setup, set_sets, inner_adapt_lpo, accuracy
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
parser.add_argument('--test-ways', type=int)
parser.add_argument('--test-shots', type=int)
parser.add_argument('--test-queries', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--meta-batch-size', type=int)
parser.add_argument('--iterations', type=int)
parser.add_argument('--inner-iters', type=int)
parser.add_argument('--order', type=bool)
parser.add_argument('--device', type=str)

args = parser.parse_args()


# Generating Tasks, initializing learners, loss, meta - optimizer
train_tasks, valid_tasks, test_tasks, learner = setup(
    args.dataset, args.root, args.n_ways, args.k_shots, args.q_shots, args.test_ways, args.test_shots, args.test_queries, 64, args.device)
learner_temp, learner_ttemp = learner
opt = optim.Adam(learner.parameters(), args.lr)
loss = nn.BCELoss(reduction='none')
# lr_scheduler = torch.optim.lr_scheduler.StepLR(
#     opt, step_size=20, gamma=0.5)
profiler = Profiler('LPO_{}_{}-way_{}-shot_{}-queries'.format(
    args.dataset, args.n_ways, args.k_shots, args.q_shots))

## Training ##
for iteration in tqdm.tqdm(range(args.iterations)):

    meta_train_loss = []
    meta_valid_loss = []
    meta_train_acc = []
    meta_valid_acc = []
    learner.train()

    for batch in range(args.meta_batch_size):
        opt.zero_grad()
        ttask = train_tasks.sample()
        support, y_support, queries, qs, y_queries, queries_labels = set_sets(
            ttask, args.n_ways, args.k_shots, args.q_shots, args.device)

        # Running inner adaptation loop
        for i in range(args.inner_iters):
            evaluation_loss, query_preds = inner_adapt_lpo(
                support, y_support, qs, y_queries, learner, loss, args.n_ways, args.k_shots, args.q_shots)
            evaluation_loss.backward()
            opt.step()

        meta_train_loss.append(evaluation_loss)
        evaluation_accuracy = accuracy(query_preds[::args.n_ways,], queries_labels)  # fig this
        meta_train_acc.append(evaluation_accuracy)
        
    # lr_scheduler.step()

    #learner.eval()
    for i, vtask in enumerate(valid_tasks):

        learner_temp_state = copy.deepcopy(learner.state_dict())
        learner_temp.load_state_dict(learner_temp_state)
        opt_temp = optim.Adam(learner_temp.parameters(), args.lr)
        support, y_support, queries, qs, y_queries, queries_labels = set_sets(
            vtask, args.n_ways, args.k_shots, args.q_shots, args.device)

        opt_temp.zero_grad()
        # Running inner adaptation loop
        for i in range(args.inner_iters):
            validation_loss, query_preds = inner_adapt_lpo(
                support, y_support, qs, y_queries, learner_temp, loss, args.n_ways, args.k_shots, args.q_shots)
            validation_loss.backward
            opt_temp.step()

        meta_valid_loss.append(validation_loss.item())
        validation_accuracy = accuracy(query_preds[::args.n_ways,], queries_labels)
        meta_valid_acc.append(validation_accuracy.item())

    profiler.log(row=[np.array(meta_train_acc).mean(), np.array(meta_train_acc).std(), np.array(meta_train_loss).mean(), np.array(meta_train_loss).std(), np.array(
        meta_valid_acc).mean(), np.array(
        meta_valid_acc).std(), np.array(
        meta_valid_loss).mean(), np.array(
        meta_valid_loss).std()])

    if (iteration % 10 == 0):
        print('Meta Train Accuracy: {:.4f} +- {:.4f}'.format(
            np.array(meta_train_acc).mean(), np.array(meta_train_acc).std()))
        print('Meta Valid Accuracy: {:.4f} +- {:.4f}'.format(
            np.array(meta_valid_acc).mean(), np.array(meta_valid_acc).std()))


## Testing ##
prof_test = Profiler('ProNets_test_{}_{}-way_{}-shot_{}-queries'.format(
    args.dataset, args.test_ways, args.test_shots, args.test_queries))
print('Testing on held out classes')
for i, tetask in enumerate(test_tasks):

    learner_ttemp = copy.deepcopy(learner.state_dict())
    learner_ttemp.load_state_dict(learner_ttemp)
    opt_ttemp = optim.Adam(learner_ttemp.parameters(), args.lr)
    opt_ttemp.zero_grad()
    meta_test_acc = []
    meta_test_loss = []

    support, y_support, queries, qs, y_queries, queries_labels = set_sets(
            vtask, args.n_ways, args.k_shots, args.q_shots, args.device)

    # Running inner adaptation loop
    for i in range(args.inner_iters):
        test_loss, query_preds = inner_adapt_lpo(
            support, y_support, qs, y_queries, learner_temp, loss, args.n_ways, args.k_shots, args.q_shots)
        test_loss.backward
        opt_ttemp.step()

    meta_test_loss.append(test_loss.item())
    test_accuracy = accuracy(query_preds[::args.n_ways,], queries_labels)
    meta_test_acc.append(test_accuracy.item())
    prof_test.log(row=[np.array(meta_test_acc).mean(), np.array(meta_test_acc).std(
    ), np.array(meta_test_loss).mean(), np.array(meta_test_loss).std()])
    print('Meta Test Accuracy', np.array(meta_test_acc).mean(),
          '+-', np.array(meta_test_acc).std())
