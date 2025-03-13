from easydict import EasyDict
import yaml
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset

def load_config(file_path = "config/pretrain.yml"):
    with open(file_path, "r") as f:
        dict_config = yaml.safe_load(f)
        config = EasyDict(dict_config)
    return config

def load_positional_encoder(ckpt_path):
    from net.lightning_model import Lightning_WavePE_AutoEncoder
    #from net.model import WavePE_AutoEncoder
    #checkpoint = torch.load(ckpt_path)
    autoencoder = Lightning_WavePE_AutoEncoder.load_from_checkpoint(ckpt_path, map_location = "cpu")
    return autoencoder.net.encoder

def load_dataloader(config, transform = None):
    if config.dataset_name in ['Peptides-func', 'Peptides-struct']:
        train_dataset = LRGBDataset("data/", name = config.dataset_name, split = "train", pre_transform=transform)
        valid_dataset = LRGBDataset("data/", name = config.dataset_name, split = "val",   pre_transform=transform)
        test_dataset = LRGBDataset("data/", name = config.dataset_name,  split = "test",  pre_transform=transform)
        #num_task = train_dataset.num_classes
        num_task = 10 if "func" in config.dataset_name else 11
        task_type = 'multilabel classification' if 'func' in config.dataset_name else "regression"
        eval_metric = "ap" if "func" in config.dataset_name else "mae"
        split_idx = None
    elif config.dataset_name == "zinc":
        from torch_geometric.datasets import ZINC
        root_dir = "/cm/shared/khangnn4/WavePE/data/zinc"
        train_dataset = ZINC(root_dir, subset = True, split = "train", pre_transform=transform)
        valid_dataset = ZINC(root_dir, subset = True, split = "val", pre_transform=transform)
        test_dataset = ZINC(root_dir, subset = True, split = "test", pre_transform=transform)
        num_task = 1
        task_type = "zinc_regression"
        eval_metric = "mae"
    elif 'ogbg' in config.dataset_name.split("-"):
        dataset = PygGraphPropPredDataset(name = config.dataset_name, transform=transform)
        split_idx = dataset.get_idx_split()
        train_dataset = dataset[split_idx['train']]
        valid_dataset = dataset[split_idx['valid']]
        test_dataset = dataset[split_idx['test']]
        num_task = train_dataset.num_tasks
        task_type = train_dataset.task_type
        eval_metric = dataset.eval_metric
    
    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True, num_workers = 6)
    valid_loader = DataLoader(valid_dataset, batch_size = config.val_batch_size, shuffle = False, num_workers = 6)
    test_loader  = DataLoader(test_dataset, batch_size = config.val_batch_size, shuffle = False, num_workers = 6)

    return train_loader, valid_loader, test_loader, num_task, task_type, eval_metric, split_idx

