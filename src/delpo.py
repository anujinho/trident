import argparse

import numpy as np
import tqdm
from torch import nn, optim

from src.zoo.delpo_utils import setup, inner_adapt_delpo
from src.utils2 import Profiler

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--root', type=str)
parser.add_argument('--n-ways', type=int)
parser.add_argument('--k-shots', type=int)
parser.add_argument('--q-shots', type=int)
parser.add_argument('--inner-adapt-steps-train', type=int)
parser.add_argument('--inner-adapt-steps-test', type=int)
parser.add_argument('--inner-lr', type=float)
parser.add_argument('--meta-lr', type=float)
parser.add_argument('--meta-batch-size', type=int)
parser.add_argument('--iterations', type=int)
parser.add_argument('--order', type=str)
parser.add_argument('--device', type=str)
parser.add_argument('--download', type=str)

args = parser.parse_args()
if args.order == 'True':
    args.order = True
elif args.order == 'False':
    args.order = False

if args.download == 'True':
    args.download = True
elif args.download == 'False':
    args.download = False


# Generating Tasks, initializing learners, loss, meta - optimizer
train_tasks, valid_tasks, test_tasks, learner = setup(
    args.dataset, args.root, args.n_ways, args.k_shots, args.q_shots, args.order, args.inner_lr, args.device, download=args.download)
opt = optim.Adam(learner.parameters(), args.meta_lr)
reconst_loss = nn.MSELoss(reduction='none')
if args.order == False:
    profiler = Profiler('DELPO_{}_{}-way_{}-shot_{}-queries'.format(args.dataset,
                        args.n_ways, args.k_shots, args.q_shots))
    prof_test = Profiler('DELPO_test_{}_{}-way_{}-shot_{}-queries'.format(
        args.dataset, args.n_ways, args.k_shots, args.q_shots))
elif args.order == True:
    profiler = Profiler('FO-DELPO_{}_{}-way_{}-shot_{}-queries'.format(
        args.dataset, args.n_ways, args.k_shots, args.q_shots))
    prof_test = Profiler('FO-DELPO_test_{}_{}-way_{}-shot_{}-queries'.format(
        args.dataset, args.n_ways, args.k_shots, args.q_shots))


## Training ##
for iter in tqdm.tqdm(range(args.iterations)):
    opt.zero_grad()
    meta_train_losses = []
    meta_valid_losses = []
    meta_train_acc = []
    meta_valid_acc = []

    for batch in range(args.meta_batch_size):
        ttask = train_tasks.sample()
        model = learner.clone()
        evaluation_loss, evaluation_accuracy = inner_adapt_delpo(
            ttask, reconst_loss, model, args.n_ways, args.k_shots, args.q_shots, args.inner_adapt_steps_train, args.device)
        evaluation_loss[0].backward()
        meta_train_losses.append([l.item() for l in evaluation_loss])
        meta_train_acc.append(evaluation_accuracy.item())

    vtask = valid_tasks.sample()
    model = learner.clone()
    validation_loss, validation_accuracy = inner_adapt_delpo(
        vtask, reconst_loss, model, args.n_ways, args.k_shots, args.q_shots, args.inner_adapt_steps_train, args.device)
    meta_valid_losses.append([l.item() for l in validation_loss])
    meta_valid_acc.append(validation_accuracy.item())

    profiler.log([np.array(meta_train_acc).mean(), np.array(meta_train_losses[::5]).mean(), np.array(meta_train_losses[1::5]).mean(),
    np.array(meta_train_losses[2::5]).mean(), np.array(meta_train_losses[3::5]).mean(), np.array(meta_train_losses[4::5]).mean(),
    np.array(meta_train_acc).std(), np.array(meta_train_losses[::5]).std(), np.array(meta_train_losses[1::5]).std(),
    np.array(meta_train_losses[2::5]).std(), np.array(meta_train_losses[3::5]).std(), np.array(meta_train_losses[4::5]).std(),
    np.array(meta_valid_acc).mean(), np.array(meta_valid_losses[::5]).mean(), np.array(meta_valid_losses[1::5]).mean(),
    np.array(meta_valid_losses[2::5]).mean(), np.array(meta_valid_losses[3::5]).mean(), np.array(meta_valid_losses[4::5]).mean()])

    if (iter % 500 == 0):
        print('Meta Train Accuracy: {:.4f} +- {:.4f}'.format(
            np.array(meta_train_acc).mean(), np.array(meta_train_acc).std()))
        print('Meta Valid Accuracy: {:.4f} +- {:.4f}'.format(
            np.array(meta_valid_acc).mean(), np.array(meta_valid_acc).std()))

    for p in learner.parameters():
        p.grad.data.mul_(1.0 / args.meta_batch_size)
    opt.step()

#torch.save(learner, f='../repro')

## Testing ##
print('Testing on held out classes')

for i, tetask in enumerate(test_tasks):
    meta_test_acc = []
    meta_test_losses = []
    model = learner.clone()
    #tetask = test_tasks.sample()
    evaluation_loss, evaluation_accuracy = inner_adapt_delpo(
        tetask, reconst_loss, model, args.n_ways, args.k_shots, args.q_shots, args.inner_adapt_steps_test, args.device)
    meta_test_losses.append([l.item() for l in evaluation_loss])
    meta_test_acc.append(evaluation_accuracy.item())
    prof_test.log(row=[np.array(meta_test_acc).mean(), np.array(meta_test_losses[::5]).mean(), np.array(meta_test_losses[1::5]).mean(),
    np.array(meta_test_losses[2::5]).mean(), np.array(meta_test_losses[3::5]).mean(), np.array(meta_test_losses[4::5]).mean(),
    np.array(meta_test_acc).std(), np.array(meta_test_losses[::5]).std(), np.array(meta_test_losses[1::5]).std(),
    np.array(meta_test_losses[2::5]).std(), np.array(meta_test_losses[3::5]).std(), np.array(meta_test_losses[4::5]).std()])
    print('Meta Test Accuracy', np.array(meta_test_acc).mean(),
          '+-', np.array(meta_test_acc).std())
