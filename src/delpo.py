import os
import argparse
import json

#import numpy as np
import tqdm
import torch
from torch import nn, optim

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
parser.add_argument('--task-adapt', type=str)
parser.add_argument('--task-adapt-fn', type=str)
parser.add_argument('--experiment', type=str)
parser.add_argument('--order', type=str)
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

if args.task_adapt == 'True':
    args.task_adapt = True
elif args.task_adapt == 'False':
    args.task_adapt = False

# wandb.config.update(args)

# Generating Tasks, initializing learners, loss, meta - optimizer and profilers
train_tasks, valid_tasks, test_tasks, learner = setup(
    args.dataset, args.root, args.n_ways, args.k_shots, args.q_shots, args.order, args.inner_lr, args.device, download=args.download, task_adapt=args.task_adapt, task_adapt_fn=args.task_adapt_fn, args=args)
opt = optim.Adam(learner.parameters(), args.meta_lr)
reconst_loss = nn.MSELoss(reduction='none')

if args.resume == 'Yes':
    learner = learner.to('cpu')
    learner = torch.load('{}/model_{}.pt'.format(args.model_path, args.iter_resume))
    learner = learner.to(args.device)
    opt.load_state_dict(torch.load('{}/opt_{}.pt'.format(args.model_path, args.iter_resume)))
    start = args.iter_resume + 1

else:
    start = 0

if args.order == False:
    profiler = Profiler('DELPO_{}_{}-way_{}-shot_{}-queries'.format(args.dataset,
                        args.n_ways, args.k_shots, args.q_shots), args.experiment)

elif args.order == True:
    profiler = Profiler('FO-DELPO_{}_{}-way_{}-shot_{}-queries'.format(
        args.dataset, args.n_ways, args.k_shots, args.q_shots), args.experiment)


## Training ##
for iter in tqdm.tqdm(range(start, args.iterations)):
    opt.zero_grad()
    batch_losses = []

    for batch in range(args.meta_batch_size):
        ttask = train_tasks.sample()
        model = learner.clone()
        if (iter % 500 == 0) & (batch == 0):
            evaluation_loss, evaluation_accuracy, reconst_img, query_imgs, mu_l, log_var_l, mu_s, log_var_s = inner_adapt_delpo(
                ttask, reconst_loss, model, args.n_ways, args.k_shots, args.q_shots, args.inner_adapt_steps_train, args.device, True, args)

            # Logging train-task images and latents
            di = {"reconst_examples": reconst_img, "gt_examples": query_imgs}
            dl = {"label_latents": [mu_l, log_var_l],
                  "style_latents": [mu_s, log_var_s]}
            profiler.log_data(di, iter, 'images', 'train')
            profiler.log_data(dl, iter, 'latents', 'train')

        else:
            evaluation_loss, evaluation_accuracy = inner_adapt_delpo(
                ttask, reconst_loss, model, args.n_ways, args.k_shots, args.q_shots, args.inner_adapt_steps_train, args.device, False, args)

        evaluation_loss['elbo'].backward()

        # Logging per train-task losses and accuracies
        tmp = [(iter*args.meta_batch_size)+batch, evaluation_accuracy.item()]
        tmp = tmp + [a.item() for a in evaluation_loss.values()]
        batch_losses.append(tmp)

    #     wandb.log(dict({f"train/{key}": loss.item() for _, (key, loss) in enumerate(evaluation_loss.items())},
    #               **{'train/accuracies': evaluation_accuracy.item(), 'train/task': (iter*args.meta_batch_size)+batch}))

    # rimages = wandb.Image(reconst_img, caption="Reconstructed Query Images")
    # qimages = wandb.Image(query_imgs, caption="Query Images")
    # wandb.log({"reconst_examples": rimages, "gt_examples": qimages})

    vtask = valid_tasks.sample()
    model = learner.clone()
    if iter % 500 == 0:
        validation_loss, validation_accuracy, reconst_img, query_imgs, mu_l, log_var_l, mu_s, log_var_s = inner_adapt_delpo(
            vtask, reconst_loss, model, args.n_ways, args.k_shots, args.q_shots, args.inner_adapt_steps_train, args.device, True, args)
        # Logging valid-task images and latents
        di = {"reconst_examples": reconst_img, "gt_examples": query_imgs}
        dl = {"label_latents": [mu_l, log_var_l],
              "style_latents": [mu_s, log_var_s]}
        profiler.log_data(di, iter, 'images', 'valid')
        profiler.log_data(dl, iter, 'latents', 'valid')

    else:
        validation_loss, validation_accuracy = inner_adapt_delpo(
            vtask, reconst_loss, model, args.n_ways, args.k_shots, args.q_shots, args.inner_adapt_steps_train, args.device, False, args)

    # Logging per validation-task losses and accuracies
    tmp = [iter, validation_accuracy.item()]
    tmp = tmp + [a.item() for a in validation_loss.values()]

    # wandb.log(dict({f"valid/{key}": loss.item() for _, (key, loss) in enumerate(validation_loss.items())},
    #           **{'valid/accuracies': validation_accuracy.item(), 'valid/task': iter}))

    # Meta backpropagation of gradients
    for p in learner.parameters():
        p.grad.data.mul_(1.0 / args.meta_batch_size)
    opt.step()

    # Saving the Logs
    profiler.log_csv(batch_losses, 'train')
    profiler.log_csv(tmp, 'valid')

    # Checkpointing the learner
    if iter % 500 == 0:
        learner = learner.to('cpu')
        profiler.log_model(learner, opt, iter)
        learner = learner.to(args.device)
    else:
        continue

profiler.log_model(learner, opt, 'last')
profiler.log_model(learner, opt, iter)

## Testing ##
print('Testing on held out classes')
# offloading unused tensors from the gpu
del evaluation_loss, evaluation_accuracy, validation_loss, validation_accuracy, reconst_img, query_imgs, mu_l, log_var_l, mu_s, log_var_s
learner.to(args.device)

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
