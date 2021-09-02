import random

import numpy as np
from numpy.core.numeric import ones_like
import torch
from torch.nn import functional as F
#from torch.nn.modules.loss import CrossEntropyLoss, KLDivLoss
from data.taskers import gen_tasks
from PIL.Image import LANCZOS
from torchvision import transforms
#from torch.distributions.multivariate_normal import MultivariateNormal

from src.zoo.archs import CVAE, LVAE, ResNet, BasicBlock


def setup(dataset, root, n_ways, k_shots, q_shots, test_ways, test_shots, test_queries, device):
    if dataset == 'omniglot':
        channels = 1
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
                               n_ways=test_ways, k_shots=test_shots, q_shots=test_queries, classes=classes[1200:], num_tasks=200)

    elif dataset == 'miniimagenet':
        channels = 3
        # Generating tasks and model according to the MAML implementation for MiniImageNet
        train_tasks = gen_tasks(dataset, root, mode='train',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)
        valid_tasks = gen_tasks(dataset, root, mode='validation',
                                n_ways=test_ways, k_shots=test_shots, q_shots=test_queries, num_tasks=200)
        test_tasks = gen_tasks(dataset, root, mode='test',
                               n_ways=test_ways, k_shots=test_shots, q_shots=test_queries, num_tasks=200)

#     train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)
#     valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)
#     test_loader = DataLoader(test_tasks, pin_memory=True, shuffle=True)


        learner = CVAE(in_channels=channels, y_shape=n_ways,
                    base_channels=32, latent_dim=64)
        learner = learner.to(device)
        learner = LVAE(in_dims=512, y_shape=n_ways, latent_dim=64)
        learner = learner.to(device)

        embedder = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=64, remove_linear=True)
        checkpoint = torch.load('/home/nfs/anujsingh/meta_lrng/files/checkpoint.pth.tar')
        model_dict = embedder.state_dict()
        params = checkpoint['state_dict']
        params = {k: v for k, v in params.items() if k in model_dict}
        model_dict.update(params)
        embedder.load_state_dict(model_dict)
        embedder.to(device)
        for p in embedder.parameters():
            p.requires_grad = False

    return train_tasks, valid_tasks, test_tasks, learner, learner, embedder


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def proto_distr(mus, log_vars, n, k, type):
    if type == 'average':
        mu_p = mus.view(n, k, -1).mean(dim=1)
        var_p = torch.exp(log_vars).view(n, k, -1).mean(dim=1)/(k)

    elif type == 'precision_weighted':
        var_p = torch.exp(log_vars).view(n, k, -1)**(-1)
        mu_p = torch.mul(mus.view(n, k, -1), var_p).sum(dim=1)
        var_p = var_p.sum(dim=1)**(-1)
        mu_p = torch.mul(mu_p, var_p)
        var_p = var_p/(k**(-1))

    return mu_p, var_p


def classify(mu_p, var_p, mu_datums):
    a = mu_datums.shape[0]
    b = mu_p.shape[0]

    # logits = MultivariateNormal(mu_p, torch.diag_embed(var_p)).log_prob(
    #     mu_datums.unsqueeze(1).expand(a, b, -1))
    logits = - 0.5 * np.log(2 * np.pi) - torch.log(var_p).unsqueeze(0).expand(a, b, -1) / 2 - (mu_datums.unsqueeze(1).expand(
        a, b, -1) - mu_p.unsqueeze(0).expand(a, b, -1))**2 / (2 * var_p.unsqueeze(0).expand(a, b, -1))
    return torch.sum(logits, dim=-1)


def kl_div(mus, log_vars):
    return - 0.5 * (1 + log_vars - mus**2 - torch.exp(log_vars)).sum(dim=1)


def set_sets(task, n_ways, k_shots, q_shots, embedder, device):
    """ Creating support and reshaped query sets """

    data, labels = task
    data, labels = data.to(device), labels.to(device)
    #data, labels = data.squeeze(0), labels.squeeze(0)
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)
    total = n_ways * (k_shots + q_shots)
    queries_index = np.zeros(total)

    # Extracting the query datums from the entire task set
    for offset in range(n_ways):
        queries_index[np.random.choice(
            k_shots+q_shots, q_shots, replace=False) + ((k_shots + q_shots)*offset)] = True
    support = data[np.where(queries_index == 0)]
    support_labels = labels[np.where(queries_index == 0)]
    queries = data[np.where(queries_index == 1)]
    queries_labels = labels[np.where(queries_index == 1)]

    y_support = F.one_hot(support_labels, num_classes=n_ways)
    y_queries = torch.tensor(range(n_ways))
    y_queries = y_queries.repeat(n_ways*q_shots)
    y_queries = F.one_hot(y_queries, num_classes=n_ways)
    qs = queries.repeat_interleave(n_ways, dim=0)

    support, qs = embedder(support), embedder(qs)

    return support, y_support.to(device), queries, qs, y_queries.to(device), queries_labels


def inner_adapt_lpo(support, y_support, qs, y_queries, learner, reconstruction_loss, n_ways, k_shots, q_shots, alpha_dec, beta):
    """ Performing Inference by minimizing (data, label) -log-likelihood over support images and (data) -log-likelihood over query images """
    # Forward pass on the Support datums
    support_cap, support_mu, support_log_var = learner(support, y_support)

    # Building Prototypical distributions
    proto_mu, proto_var = proto_distr(
        support_mu, support_log_var, n_ways, k_shots, 'average')

    # Forward pass on the Query datums
    queries_cap, queries_mu, queries_log_var = learner(qs, y_queries)

    support_logits = classify(
        mu_p=proto_mu, var_p=proto_var, mu_datums=support_mu)
    queries_logits = classify(
        mu_p=proto_mu, var_p=proto_var, mu_datums=queries_mu)

    # adding up the losses
    ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    L_support = -reconstruction_loss(support_cap, support).view(support.shape[0], -1).mean(dim=1) - ce_loss(
        F.softmax(torch.ones_like(y_support).float(), dim=1), torch.argmax(y_support, dim=1)) - kl_div(support_mu, support_log_var)  # = -L(x_s, y_s)

    L_queries = -reconstruction_loss(queries_cap, qs).view(qs.shape[0], -1).mean(dim=1) - ce_loss(
        F.softmax(torch.ones_like(y_queries).float(), dim=1), torch.argmax(y_queries, dim=1)) - kl_div(queries_mu, queries_log_var)  # = -L(x_q, y_q)

    U_queries = torch.mul(F.softmax(queries_logits, dim=1)[
                         ::n_ways, ], L_queries.view(n_ways*q_shots, n_ways)).sum(dim=1) - beta*torch.sum(torch.mul(F.softmax(queries_logits, dim=1)[
                             ::n_ways, ], torch.log(F.softmax(queries_logits, dim=1)[
                                 ::n_ways, ])), dim=1)
    alpha = alpha_dec*(q_shots/k_shots)
    J_alpha = - L_support.mean() - U_queries.mean() + alpha * \
        ce_loss(support_logits, torch.argmax(y_support, dim=1)).mean()
    J_alpha = J_alpha.mean()

    return J_alpha, F.softmax(queries_logits, dim=1)
