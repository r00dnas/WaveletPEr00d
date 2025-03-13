import os
import argparse
import torch 
from torch_geometric import seed_everything
from transform import WaveletTransform
from utils.commons import load_config, load_dataloader
from utils.train_test import train_one_epoch, eval_one_epoch

from net.graph_transformer import GraphTransformer
from net.gnn import GNN_VirtualNode, GNN, GINETransfer, GATTransfer

from ogb.graphproppred import Evaluator
import numpy as np
import json
from utils.metrics_wrapper import MetricWrapper
import transformers
from torch.optim.lr_scheduler import StepLR, ExponentialLR


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, default = "moleculenet")
    parser.add_argument("--local_gnn_type", type = str, default = "gine")
    parser.add_argument("--scheduler", type = str, default = "cosine_with_warmup")
    parser.add_argument("--warmup_steps", type = int)
    parser.add_argument("--num_epoch", type = int)
    parser.add_argument("--model_type", type = str)
    parser.add_argument("--residual",type = int, default = 0)
    parser.add_argument("--freeze", type = int, default = 0)
    parser.add_argument("--dropout", type = float, default = 0.5)
    parser.add_argument("--num_layer", type = int, default = 5)
    parser.add_argument("--atom_dim", type = int, default = 300)
    parser.add_argument("--bond_dim", type = int, default = 300)
    parser.add_argument("--lr", type = float, default = 1e-3)
    args = parser.parse_args()

    ### Random seed generator ###
    seed = torch.randint(0,50, ()).item()
    seed_everything(seed)


    pretrained_config = load_config(f"config/pretrain.yml")
    transform = WaveletTransform(pretrained_config.scales, approximation_order=pretrained_config.approximation_order, tolerance=pretrained_config.tolerance) 
    downstream_config = load_config(f"config/{args.config}.yml")
    train_loader, valid_loader, test_loader, num_task, task_type, eval_metric, split_idx = load_dataloader(downstream_config, transform)

    downstream_config.local_gnn_type = args.local_gnn_type
    downstream_config.scheduler = args.scheduler 
    downstream_config.warmup_steps = args.warmup_steps
    downstream_config.num_epoch = args.num_epoch
    downstream_config.model_type = args.model_type
    downstream_config.residual = bool(args.residual)
    downstream_config.freeze = bool(args.freeze)
    downstream_config.num_layer = args.num_layer
    downstream_config.atom_dim = args.atom_dim
    downstream_config.bond_dim = args.bond_dim
    downstream_config.dropout = args.dropout
    downstream_config.lr = args.lr
    
    if downstream_config.model_type == "gnn_vn":
        model = GNN_VirtualNode(downstream_config, num_task).to(downstream_config.device)
    elif downstream_config.model_type == "gnn":
        model = GNN(downstream_config, num_task).to(downstream_config.device)
    elif downstream_config.model_type == "gine_transfer":
        model = GINETransfer(downstream_config, num_task).to(downstream_config.device)
    elif downstream_config.model_type == "gat_transfer":
        model = GATTransfer(downstream_config, num_task).to(downstream_config.device)
    else:
        model = GraphTransformer(downstream_config, num_task).to(downstream_config.device)
        #model = GINE_LSPE(downstream_config.atom_dim, downstream_config.num_layer, "add", downstream_config).to(downstream_config.device)
    device = downstream_config.device
     
    if downstream_config.dataset_name == "Peptides-func":
        name = "ogbg-molpcba" 
        evaluator = Evaluator(name)
        evaluator.num_tasks = 10
    elif downstream_config.dataset_name in ["Peptides-struct", "zinc"]:
        evaluator = MetricWrapper(metric = "mae")
        eval_metric = "mae"
    else:
        name = downstream_config.dataset_name
        evaluator = Evaluator(name)

    print(f"Task: {task_type}, Num Task: {num_task}")
    print(f"Trainable parameters: {model.num_trainable_parameters}")
    print(f"Local gnn type: ", {downstream_config.local_gnn_type})
    print(f"Scheduler used: ", {downstream_config.scheduler})

    if downstream_config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr = float(downstream_config.lr), weight_decay=float(downstream_config.weight_decay))
    else:
        downstream_config.optimizer = "adam"
        optimizer = torch.optim.Adam(model.parameters(), lr = float(downstream_config.lr), weight_decay=float(downstream_config.weight_decay))

    if downstream_config.scheduler == "reduce_on_plateau":
        mode = "max" if eval_metric in ["ap", "rocauc", "acc"] else "min"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, min_lr = 1e-6, verbose=True, factor = 0.5, patience = 10)
    elif downstream_config.scheduler == "cosine_with_warmup":
        scheduler = transformers.optimization.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=downstream_config.warmup_steps, num_training_steps=downstream_config.num_epoch)
    elif downstream_config.scheduler == "step_lr":
        scheduler = StepLR(optimizer, step_size=20, gamma=0.3)
    elif downstream_config.scheduler == "exp_lr":
        scheduler = ExponentialLR(optimizer, gamma=0.9, last_epoch=downstream_config.num_epoch)
    else:
        scheduler = None
    
    train_loss_curve = []
    valid_curve = []
    test_curve = []
    best_val_epoch = 0
    for epoch in range(downstream_config.num_epoch):
        train_loss = train_one_epoch(model, device, train_loader, optimizer, task_type, downstream_config, False)

        valid_perf = eval_one_epoch(model, device, valid_loader, evaluator, False)
        test_perf = eval_one_epoch(model, device, test_loader, evaluator, False)

        train_loss_curve.append(train_loss)
        valid_curve.append(valid_perf[eval_metric])
        test_curve.append(test_perf[eval_metric])
        
        if downstream_config.scheduler == "reduce_on_plateau":
            scheduler.step(valid_perf[eval_metric])
        elif downstream_config.scheduler == "cosine_with_warmup":
            scheduler.step()
        elif downstream_config.scheduler == "step_lr":
            scheduler.step()
        else:
            pass
        
        print({'Epoch': epoch, 'Train': train_loss, 'Validation': valid_perf, 'Test': test_perf, "Test at best val": test_curve[best_val_epoch], "Best val": valid_curve[best_val_epoch]})
        print()

        if "classification" in task_type:
            best_val_epoch = np.argmax(np.array(valid_curve))
            best_train = min(train_loss_curve)
        else:
            best_val_epoch = np.argmin(np.array(valid_curve))
            best_train = min(train_loss_curve)
        
        if epoch == downstream_config.num_epoch - 1:
            if test_perf[eval_metric] > test_curve[best_val_epoch]:
                test_curve[best_val_epoch] = test_perf[eval_metric]

    print("Finish training")
    print(f"Best validation score: {valid_curve[best_val_epoch]}")
    print(f"Test score: {test_curve[best_val_epoch]}")

    
    os.makedirs(f"logs/{downstream_config.dataset_name}", exist_ok=True)
    save_path = os.path.join(f"logs/{downstream_config.dataset_name}", f"{args.model_type}_{downstream_config.local_gnn_type}_results.json")
    try:
        with open(save_path, "r") as file_:
            content = json.load(file_)
    except:
        content = {}
    content.update({
        seed : { "model_type" : downstream_config.model_type,
                #"local_gnn_type":downstream_config.local_gnn_type,
                "valid_score" : valid_curve[best_val_epoch], 
                "test_score" : test_curve[best_val_epoch], 
                "best_train_loss": train_loss_curve[best_val_epoch], 
                "trainable_params" : model.num_trainable_parameters, 
                "pretrain_path": downstream_config.ckpt_pos_encoder_path, 
                "use_residual" : downstream_config.residual, 
                "warmup_step" : downstream_config.warmup_steps, 
                "idx": len(content) + 1} 
    })

    with open(save_path, "w") as file_:
        json.dump(content, file_, indent=4)
