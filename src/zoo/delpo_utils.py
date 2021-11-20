import random

import learn2learn as l2l
import numpy as np
import torch
from torch.nn import functional as F
from data.taskers import gen_tasks
from PIL.Image import LANCZOS
from torchvision import transforms

from src.zoo.archs import CCVAE, ResNet12Backbone


def setup(dataset, root, n_ways, k_shots, q_shots, order, inner_lr, device, download, task_adapt, task_adapt_fn, args):
    if dataset == 'omniglot':
        image_trans = transforms.Compose([transforms.Resize(
            28, interpolation=LANCZOS), transforms.ToTensor(), lambda x: 1-x])
        classes = list(range(1623))  # Total classes in Omniglot
        random.shuffle(classes)
        # Generating tasks and model according to the MAML implementation for Omniglot
        train_tasks = gen_tasks(dataset, root, image_transforms=image_trans,
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, classes=classes[:1100], num_tasks=20000)
        valid_tasks = gen_tasks(dataset, root, image_transforms=image_trans, n_ways=n_ways,
                                k_shots=k_shots, q_shots=q_shots, classes=classes[1100:1200], num_tasks=200)
        test_tasks = gen_tasks(dataset, root, image_transforms=image_trans,
                               n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, classes=classes[1200:], num_tasks=600)
        learner = CCVAE(in_channels=1, base_channels=64,
                        n_ways=n_ways, dataset='omniglot', task_adapt=task_adapt, task_adapt_fn=task_adapt_fn, args=args)

    elif (dataset == 'miniimagenet'):
        # Generating tasks and model according to the MAML implementation for MiniImageNet
        train_tasks = gen_tasks(dataset, root, download=download, mode='train',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)
        valid_tasks = gen_tasks(dataset, root, download=download, mode='validation',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)
        test_tasks = gen_tasks(dataset, root, download=download, mode='test',
                               n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, num_tasks=600)
        learner = CCVAE(in_channels=3, base_channels=32,
                        n_ways=n_ways, dataset='mini_imagenet', task_adapt=task_adapt, task_adapt_fn=task_adapt_fn, args=args)

    elif (dataset == 'tiered'):
        image_trans = transforms.Compose([transforms.ToTensor()])
        train_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='train',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)  # , num_tasks=50000)
        valid_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='validation',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)  # , num_tasks=10000)
        test_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='test',
                               n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, num_tasks=2000)
        learner = CCVAE(in_channels=3, base_channels=32,
                        n_ways=n_ways, dataset='tiered', task_adapt=task_adapt, task_adapt_fn=task_adapt_fn, args=args)

    elif dataset == 'cifarfs':
        image_trans = transforms.Compose([transforms.ToTensor()])
        train_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='train',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)
        valid_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='validation',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)
        test_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='test',
                               n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, num_tasks=600)
        learner = CCVAE(in_channels=3, base_channels=64,
                        n_ways=n_ways, dataset='cifarfs', task_adapt=task_adapt, task_adapt_fn=task_adapt_fn, args=args)

    learner = learner.to(device)
    learner = l2l.algorithms.MAML(learner, first_order=order, lr=inner_lr)

    # Init the Backbone
    if args.pretrained[0] == True:
        backbone = ResNet12Backbone(
            args, avg_pool=True if args.pretrained[2] == 640 else False)  # F => 16000; T => 640
        weights = torch.load(args.pretrained[1])
        backbone.load_state_dict(weights)
        backbone.to(args.device)
        # Freeze the backbone
        for p in backbone.parameters():
            p.requires_grad = False
    else:
        backbone = 'None'

    return train_tasks, valid_tasks, test_tasks, learner, backbone


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def kl_div(mus, log_vars):
    return - 0.5 * (1 + log_vars - mus**2 - torch.exp(log_vars)).sum(dim=1)


def loss(reconst_loss: object, reconst_image, image, logits, labels, mu_s, log_var_s, mu_l, log_var_l, wt_ce=1e2, klwt=False, rec_wt=1e-2, beta_l=1, beta_s=1):
    kl_div_s = kl_div(mu_s, log_var_s).mean()
    kl_div_l = kl_div(mu_l, log_var_l).mean()
    if klwt:
        kl_wt = mu_s.shape[-1] / (image.shape[-1] *
                                  image.shape[-2] * image.shape[-3])
    else:
        kl_wt = 1

    ce_loss = torch.nn.CrossEntropyLoss()
    classification_loss = ce_loss(logits, labels)
    rec_loss = reconst_loss(reconst_image, image)
    rec_loss = rec_loss.view(rec_loss.shape[0], -1).sum(dim=-1).mean()

    L = wt_ce*classification_loss + beta_l*kl_wt*kl_div_l + \
        rec_wt*rec_loss + beta_s*kl_wt*kl_div_s  # -log p(x,y)

    losses = {'elbo': L, 'label_kl': kl_div_l, 'style_kl': kl_div_s,
              'reconstruction_loss': rec_loss, 'classification_loss': classification_loss}

    return losses


def inner_adapt_delpo(task, reconst_loss, learner, n_ways, k_shots, q_shots, adapt_steps, device, log_data: bool, args, backbone):
    data, labels = task
    if args.dataset == 'miniimagenet':
        data, labels = data.to(device) / 255.0, labels.to(device)
    elif (args.dataset == 'omniglot') or (args.dataset == 'cifarfs') or (args.dataset == 'tiered'):
        data, labels = data.to(device), labels.to(device)
    total = n_ways * (k_shots + q_shots)
    queries_index = np.zeros(total)

    # Extracting the evaluation datums from the entire task set, for the meta gradient calculation
    for offset in range(n_ways):
        queries_index[np.random.choice(
            k_shots+q_shots, q_shots, replace=False) + ((k_shots + q_shots)*offset)] = True
    support = data[np.where(queries_index == 0)]
    support_labels = labels[np.where(queries_index == 0)]
    queries = data[np.where(queries_index == 1)]
    queries_labels = labels[np.where(queries_index == 1)]

    if args.pretrained[0] == True:
        support_ext = backbone(support)
        queries_ext = backbone(queries)

    # Inner adapt step
    if args.pretrained[0] == True:
        for _ in range(adapt_steps):
            if args.task_adapt:
                reconst_image, logits, mu_l, log_var_l, mu_s, log_var_s = learner([torch.cat(
                    [support_ext, queries_ext], dim=0), torch.cat([support, queries], dim=0)], 'inner')
            else:
                reconst_image, logits, mu_l, log_var_l, mu_s, log_var_s = learner(
                    [support_ext, support], 'inner')
            adapt_loss = loss(reconst_loss, reconst_image, support,
                              logits, support_labels, mu_s, log_var_s, mu_l, log_var_l, args.wt_ce, args.klwt, args.rec_wt, args.beta_l, args.beta_s)
            learner.adapt(adapt_loss['elbo'])

        if args.task_adapt:
            reconst_image, logits, mu_l, log_var_l, mu_s, log_var_s = learner([torch.cat(
                [support_ext, queries_ext], dim=0), torch.cat([support, queries], dim=0)], 'outer')
        else:
            reconst_image, logits, mu_l, log_var_l, mu_s, log_var_s = learner(
                [queries_ext, queries], 'outer')

    if args.pretrained[0] == False:
        for _ in range(adapt_steps):
            if args.task_adapt:
                reconst_image, logits, mu_l, log_var_l, mu_s, log_var_s = learner(
                    torch.cat([support, queries], dim=0), 'inner')
            else:
                reconst_image, logits, mu_l, log_var_l, mu_s, log_var_s = learner(
                    support, 'inner')
            adapt_loss = loss(reconst_loss, reconst_image, support,
                              logits, support_labels, mu_s, log_var_s, mu_l, log_var_l, args.wt_ce, args.klwt, args.rec_wt, args.beta_l, args.beta_s)
            learner.adapt(adapt_loss['elbo'])

        if args.task_adapt:
            reconst_image, logits, mu_l, log_var_l, mu_s, log_var_s = learner(
                torch.cat([support, queries], dim=0), 'outer')
        else:
            reconst_image, logits, mu_l, log_var_l, mu_s, log_var_s = learner(
                queries, 'outer')

    eval_loss = loss(reconst_loss, reconst_image, queries,
                     logits, queries_labels, mu_s, log_var_s, mu_l, log_var_l, args.wt_ce, args.klwt, args.rec_wt, args.beta_l, args.beta_s)
    eval_acc = accuracy(F.softmax(logits, dim=1), queries_labels)

    if log_data:
        return eval_loss, eval_acc, reconst_image.detach().to('cpu'), queries.detach().to('cpu'), mu_l.detach().to('cpu'), log_var_l.detach().to('cpu'), mu_s.detach().to('cpu'), log_var_s.detach().to('cpu')
    else:
        return eval_loss, eval_acc
