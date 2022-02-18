import numpy as np
import torch
from torch.nn import functional as F
from data.cdfsl import crop, eurosat, isic


def setup(dataset, n_ways, k_shots, q_shots, args):
    """ Returns task-sets with 600 randomly sampled tasks for cross-domain testing """

    params   = dict(n_way = n_ways, n_support = k_shots) 
    if dataset == 'crop':
        base_datamgr = crop.SetDataManager(84, n_eposide = 600, n_query = q_shots, **params, args=args)
        taskers = base_datamgr.get_data_loader(aug = False)
    
    elif dataset == 'eurosat':
        base_datamgr = eurosat.SetDataManager(84, n_eposide = 600, n_query = q_shots, **params, args=args)
        taskers = base_datamgr.get_data_loader(aug = False)
    
    elif dataset == 'isic':
        base_datamgr = isic.SetDataManager(84, n_eposide = 600, n_query = q_shots, **params, args=args)
        taskers = base_datamgr.get_data_loader(aug = False)

    return taskers


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


def inner_adapt_delpo(task, reconst_loss, learner, n_ways, k_shots, q_shots, adapt_steps, device, log_data: bool, args):
    
    data, labels = task

    # Extracting the evaluation datums from the entire task set, for the meta gradient calculation
    support, queries = data[:, :k_shots, :, :, :].reshape(-1, 3, 84, 84), data[:, k_shots:, :, :, :].reshape(-1, 3, 84, 84)
    support_labels, queries_labels = labels[:, :k_shots].reshape(-1), labels[:, k_shots:].reshape(-1)

    # Remapping the labels to integers in [0,n_ways]
    d = dict(enumerate(np.array(labels.unique()).flatten()))
    d = {v: k for k, v in d.items()}
    support_labels, queries_labels = torch.tensor(np.vectorize(d.get)(support_labels)), torch.tensor(np.vectorize(d.get)(queries_labels))
    support, queries = support.to(device), queries.to(device)
    support_labels, queries_labels = support_labels.to(device), queries_labels.to(device)

    # Inner adapt step
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

    if log_data :
        return eval_loss, eval_acc, reconst_image.detach().to('cpu'), queries.detach().to('cpu'), mu_l.detach().to('cpu'), log_var_l.detach().to('cpu'), mu_s.detach().to('cpu'), log_var_s.detach().to('cpu'), logits.detach().to('cpu'), queries_labels.detach().to('cpu')
    else:
        return eval_loss, eval_acc
