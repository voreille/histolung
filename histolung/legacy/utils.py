from pathlib import Path
import pandas as pd
import numpy as np
from numpy.linalg import norm
import torch
from easydict import EasyDict as edict
from typing import Optional, List
import yaml
import wandb
from sklearn.metrics import accuracy_score
from histolung.legacy.datasets import Dataset_instance_MIL
from torch.utils.data import DataLoader


thispath = Path(__file__).resolve()


def get_generator_instances(csv_patches_path, preprocess, batch_size, pipeline_transform, num_workers):

    params_instance = {'batch_size': batch_size,
                    'num_workers': num_workers,
                    'pin_memory': True,
                    'shuffle': True}

    instances = Dataset_instance_MIL(csv_patches_path, pipeline_transform, preprocess)
    generator = DataLoader(instances, **params_instance)

    return generator


def accuracy_micro(y_true, y_pred):

    y_true_flatten = y_true.flatten()
    y_pred_flatten = y_pred.flatten()
    
    return accuracy_score(y_true_flatten, y_pred_flatten)

    
def accuracy_macro(y_true, y_pred):

    n_classes = 4

    acc_tot = 0.0

    for i in range(n_classes):

        acc = accuracy_score(y_true[i,:], y_pred[i,:])
        #print(acc)
        acc_tot = acc_tot + acc

    acc_tot = acc_tot/n_classes

    return acc_tot


def cosine_similarity(a, b):
    
    return np.dot(a,b)/(norm(a)*norm(b))


def edict2dict(edict_obj):
    dict_obj = {}

    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals

    return dict_obj


def initialize_wandb(
    cfg,
    outputdir,
    iterator_fold = None
):
    outputdir = Path(outputdir / "wandb")
    Path(outputdir).mkdir(exist_ok=True, parents=True)
    if iterator_fold == None:
        name=cfg.experiment_name
    else:
        name=f"{cfg.experiment_name}_{iterator_fold}"    
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.username,
        name=name,
        group=cfg.wandb.group,
        dir= outputdir,
        config=cfg,
    )
    return run 


def yaml_load(fileName):
    dict_config = None
    with open(fileName, 'r') as ymlfile:
        dict_config = edict(yaml.safe_load(ymlfile))

    return dict_config


def generate_list_instances(wsi_path):
    # read the csv with the path for patches
    pyhistdir = Path(thispath.parent.parent / "data" / "Mask_PyHIST_v2")

    good_patches_path = Path(pyhistdir /
                             wsi_path.parent.stem /
                             wsi_path.stem /
                             f"{wsi_path.stem}_ensely_filtered_paths.csv")

    good_patches_df = pd.read_csv(good_patches_path)

    return good_patches_df


def contrastive_loss(q, k, queue, temperature):

    batch_size = q.shape[0]

    # dimension is the dimension of the representation
    dimension = q.shape[1]

    # BMM stands for batch matrix multiplication
    # If mat1 is B × n × M tensor, then mat2 is B × m × P tensor,
    # Then output a B × n × P tensor.
    pos = torch.exp(torch.div(torch.bmm(q.view(batch_size, 1, dimension),
                                        k.view(batch_size, dimension, 1)).view(batch_size, 1),
                              temperature))

    # Matrix multiplication is performed between the query and the queue tensor
    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(batch_size, dimension),
                                                 torch.t(queue)), temperature)), dim=1)

    # Sum up
    denominator = neg + pos

    return torch.mean(-torch.log(torch.div(pos, denominator)))


def momentum_step(encoder, momentum_encoder, m=1):

    params_q = encoder.state_dict()
    params_k = momentum_encoder.state_dict()

    dict_params_k = dict(params_k)

    for name in params_q:
        theta_k = dict_params_k[name]
        theta_q = params_q[name].data
        dict_params_k[name].data.copy_(m * theta_k + (1-m) * theta_q)

    momentum_encoder.load_state_dict(dict_params_k)


def update_lr(epoch, actual_lr, optimizer):

    if epoch < 10:
        lr = actual_lr
    elif epoch >= 10 and epoch < 20:
        lr = actual_lr * 0.1
    elif epoch >= 20:
        lr = actual_lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_queue(queue, k, num_keys):

    len_k = k.shape[0]
    len_queue = queue.shape[0]

    new_queue = torch.cat([k, queue], dim=0)

    new_queue = new_queue[:num_keys]

    return new_queue
