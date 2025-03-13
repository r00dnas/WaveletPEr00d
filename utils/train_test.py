import torch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction = "mean")
#reg_criterion = torch.nn.L1Loss()
reg_criterion = torch.nn.MSELoss()

def train_one_epoch(model, device, loader, optimizer, task_type, config = None, is_lapse = False):
    model.train()
    total_loss = 0
    total_size = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            if is_lapse:
                pred = model(batch, batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            elif "zinc" in task_type:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled].view(-1), batch.y.to(torch.float32)[is_labeled].view(-1))
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            if config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.detach().cpu().item() * batch.num_graphs 
            total_size += batch.num_graphs
    return total_loss / total_size

def eval_one_epoch(model, device, loader, evaluator, is_lapse = False):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                if is_lapse:
                    pred = model(batch, batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    pred = model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

#### For link prediction task ###
def link_prediction_loss(y_hat, batch):
    y = batch.edge_label.float()
    device = batch.x.device
    batch_size = batch.num_graphs
    node_ptr = batch._slice_dict['x'].to(device)
    edge_ptr = batch._slice_dict['edge_label_index'].to(device)
    num_edges = edge_ptr[1:] - edge_ptr[:-1]
    # (B, N, N) -> (sum(Ei),)
    edge_batch_index = torch.arange(batch_size, device=device).repeat_interleave(num_edges)
    edge_index_offset = node_ptr[:-1].repeat_interleave(num_edges)
    edge_index = batch.edge_label_index - edge_index_offset[None, :]
    y_hat = y_hat[edge_batch_index, edge_index[0], edge_index[1]]
    return cls_criterion(y_hat, y)

def train_inductive_link_prediction(model, device, loader, optimizer):
    model.train()
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred, label = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            loss = cls_criterion(pred, label) 
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu().item() * batch.num_graphs 
            total_size += batch.num_graphs
    return total_loss / total_size

