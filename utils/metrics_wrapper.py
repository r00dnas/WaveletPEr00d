import operator as op
from copy import deepcopy
from typing import Union, Callable, Optional, Dict, Any
import warnings

import torch
from torch_geometric.data  import Data
from torchmetrics.functional import (
    accuracy,
    average_precision,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision,
    recall,
    auroc,
    mean_absolute_error,
    mean_squared_error,
)
from torchmetrics.utilities import reduce

EPS = 1e-5


class Thresholder:
    def __init__(
        self,
        threshold: float,
        operator: str = "greater",
        th_on_preds: bool = True,
        th_on_target: bool = False,
        target_to_int: bool = False,
    ):

        # Basic params
        self.threshold = threshold
        self.th_on_target = th_on_target
        self.th_on_preds = th_on_preds
        self.target_to_int = target_to_int

        # Operator can either be a string, or a callable
        if isinstance(operator, str):
            op_name = operator.lower()
            if op_name in ["greater", "gt"]:
                op_str = ">"
                operator = op.gt
            elif op_name in ["lower", "lt"]:
                op_str = "<"
                operator = op.lt
            else:
                raise ValueError(f"operator `{op_name}` not supported")
        elif callable(operator):
            op_str = operator.__name__
        elif operator is None:
            pass
        else:
            raise TypeError(f"operator must be either `str` or `callable`, "
                            f"provided: `{type(operator)}`")

        self.operator = operator
        self.op_str = op_str

    def compute(self, preds: torch.Tensor, target: torch.Tensor):
        # Apply the threshold on the predictions
        if self.th_on_preds:
            preds = self.operator(preds, self.threshold)

        # Apply the threshold on the targets
        if self.th_on_target:
            target = self.operator(target, self.threshold)

        if self.target_to_int:
            target = target.to(int)

        return preds, target

    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
        return self.compute(preds, target)

    def __repr__(self):
        r"""
        Control how the class is printed
        """

        return f"{self.op_str}{self.threshold}"


def pearsonr(preds: torch.Tensor, target: torch.Tensor,
             reduction: str = "elementwise_mean") -> torch.Tensor:
    r"""
    Computes the pearsonr correlation.

    Parameters:
        preds: estimated labels
        target: ground truth labels
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Returns:
        Tensor with the pearsonr

    !!! Example
        ``` python linenums="1"
        x = torch.tensor([0., 1, 2, 3])
        y = torch.tensor([0., 1, 2, 2])
        pearsonr(x, y)
        >>> tensor(0.9439)
        ```
    """

    preds, target = preds.to(torch.float32), target.to(torch.float32)

    shifted_x = preds - torch.mean(preds, dim=0)
    shifted_y = target - torch.mean(target, dim=0)
    sigma_x = torch.sqrt(torch.sum(shifted_x ** 2, dim=0))
    sigma_y = torch.sqrt(torch.sum(shifted_y ** 2, dim=0))

    pearson = torch.sum(shifted_x * shifted_y, dim=0) / (sigma_x * sigma_y + EPS)
    pearson = torch.clamp(pearson, min=-1, max=1)
    pearson = reduce(pearson, reduction=reduction)
    return pearson


def _get_rank(values):

    arange = torch.arange(values.shape[0],
                          dtype=values.dtype, device=values.device)

    val_sorter = torch.argsort(values, dim=0)
    val_rank = torch.empty_like(values)
    if values.ndim == 1:
        val_rank[val_sorter] = arange
    elif values.ndim == 2:
        for ii in range(val_rank.shape[1]):
            val_rank[val_sorter[:, ii], ii] = arange
    else:
        raise ValueError(f"Only supports tensors of dimensions 1 and 2, "
                         f"provided dim=`{values.ndim}`")

    return val_rank


def spearmanr(preds: torch.Tensor, target: torch.Tensor,
              reduction: str = "elementwise_mean") -> torch.Tensor:
    r"""
    Computes the spearmanr correlation.

    Parameters:
        preds: estimated labels
        target: ground truth labels
        reduction: a method to reduce metric score over labels.
            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Returns:
        Tensor with the spearmanr

    !!! Example
        x = torch.tensor([0., 1, 2, 3])
        y = torch.tensor([0., 1, 2, 1.5])
        spearmanr(x, y)
        tensor(0.8)
    """

    spearman = pearsonr(_get_rank(preds), _get_rank(target), reduction=reduction)
    return spearman


METRICS_CLASSIFICATION = {
    "accuracy": accuracy,
    "averageprecision": average_precision,
    "auroc": auroc,
    "confusionmatrix": confusion_matrix,
    "f1": f1_score,
    "fbeta": fbeta_score,
    "precisionrecallcurve": precision_recall_curve,
    "precision": precision,
    "recall": recall,
}

METRICS_REGRESSION = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "pearsonr": pearsonr,
    "spearmanr": spearmanr,
}

METRICS_DICT = deepcopy(METRICS_CLASSIFICATION)
METRICS_DICT.update(METRICS_REGRESSION)


class MetricWrapper:
    r"""
    Allows to initialize a metric from a name or Callable, and initialize the
    `Thresholder` in case the metric requires a threshold.
    """

    def __init__(
        self,
        metric: Union[str, Callable],
        threshold_kwargs: Optional[Dict[str, Any]] = None,
        target_nan_mask: Optional[Union[str, int]] = None,
        **kwargs,
    ):
        r"""
        Parameters
            metric:
                The metric to use. See `METRICS_DICT`

            threshold_kwargs:
                If `None`, no threshold is applied.
                Otherwise, we use the class `Thresholder` is initialized with the
                provided argument, and called before the `compute`

            target_nan_mask:

                - None: Do not change behaviour if there are NaNs

                - int, float: Value used to replace NaNs. For example, if `target_nan_mask==0`, then
                  all NaNs will be replaced by zeros

                - 'ignore-flatten': The Tensor will be reduced to a vector without the NaN values.

                - 'ignore-mean-label': NaNs will be ignored when computing the loss. Note that each column
                  has a different number of NaNs, so the metric will be computed separately
                  on each column, and the metric result will be averaged over all columns.
                  *This option might slowdown the computation if there are too many labels*

            kwargs:
                Other arguments to call with the metric
        """

        self.metric_name = metric
        self.metric = METRICS_DICT[metric] if isinstance(metric, str) else metric

        self.thresholder = None
        if threshold_kwargs is not None:
            self.thresholder = Thresholder(**threshold_kwargs)

        self.target_nan_mask = target_nan_mask

        self.kwargs = kwargs

    def compute(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the metric, apply the thresholder if provided, and manage the NaNs
        """

        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)

        if target.ndim == 1:
            target = target.unsqueeze(-1)

        target_nans = torch.isnan(target)

        # Threshold the prediction
        if self.thresholder is not None:
            preds, target = self.thresholder(preds, target)

        # Manage the NaNs
        if self.target_nan_mask is None:
            pass
        elif isinstance(self.target_nan_mask, (int, float)):
            target = target.clone()
            target[torch.isnan(target)] = self.target_nan_mask
        elif self.target_nan_mask == "ignore-flatten":
            target = target[~target_nans]
            preds = preds[~target_nans]
        elif self.target_nan_mask == "ignore-mean-label":
            target_list = [target[..., ii][~target_nans[..., ii]] for ii in range(target.shape[-1])]
            preds_list = [preds[..., ii][~target_nans[..., ii]] for ii in range(preds.shape[-1])]
            target = target_list
            preds = preds_list
        else:
            raise ValueError(f"Invalid option `{self.target_nan_mask}`")

        if self.target_nan_mask == "ignore-mean-label":
            warnings.filterwarnings("error")
            # Compute the metric for each column, and output nan if there's an error on a given column
            metric_val = []
            for ii in range(len(target)):
                try:
                    kwargs = self.kwargs.copy()
                    if 'cast_to_int' in kwargs and kwargs['cast_to_int']:
                        del kwargs['cast_to_int']
                        res = self.metric(preds[ii], target[ii].int(), **kwargs)
                    else:
                        res = self.metric(preds[ii], target[ii], **kwargs)
                    metric_val.append(res)
                except Exception as e:
                    # For torchmetrics.functional.auroc do not include 0 for
                    # targets that don't have positive examples. This is what
                    # the OGB evaluator does, i.e. ignore those targets.
                    # Catching the Warning risen by torchmetrics.functional.auroc
                    # already prevents the 0 to be appended, nothing else needs
                    # to be done.
                    if str(e) == 'No positive samples in targets, ' \
                                 'true positive value should be meaningless. ' \
                                 'Returning zero tensor in true positive score':
                        pass
                    else:
                        print(e)
            warnings.filterwarnings("default")

            # Average the metric
            metric_val = torch.nanmean(torch.stack(metric_val))  # PyTorch1.10+

        else:
            metric_val = self.metric(preds, target, **self.kwargs)
        return {self.metric_name: metric_val.item()}

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the metric with the method `self.compute`
        """
        return self.compute(preds, target)
    
    def eval(self, input_dict) -> torch.Tensor:
        r"""
        Compute the metric with the method `self.compute`
        """
        preds = torch.from_numpy(input_dict['y_pred'])
        target = torch.from_numpy(input_dict['y_true'])
        return self.compute(preds, target)

    def __repr__(self):
        r"""
        Control how the class is printed
        """
        full_str = f"{self.metric.__name__}"
        if self.thresholder is not None:
            full_str += f"({self.thresholder})"

        return full_str

class LinkPredictionMetric():
    def __init__(self):
        super().__init__()

    def _eval_mrr(self, y_pred_pos, y_pred_neg): 
        """ Compute Hits@k and Mean Reciprocal Rank (MRR).
        Implementation from OGB:
        https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py
        Args:
            y_pred_neg: array with shape (batch size, num_entities_neg).
            y_pred_pos: array with shape (batch size, )
        """
        y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
        argsort = torch.argsort(y_pred, dim=1, descending=True)
        rankings = torch.nonzero(argsort == 0, as_tuple=False)
        rankings = rankings[:, 1] + 1
        hits1 = (rankings <= 1).to(torch.float)
        hits3 = (rankings <= 3).to(torch.float)
        hits10 = (rankings <= 10).to(torch.float)
        mrr = 1. / rankings.to(torch.float)
        return {'hits@1': hits1, 'hits@3': hits3, 'hits@10': hits10, 'mrr': mrr}

    @torch.no_grad()
    def _eval(self, pred, data):
        pred = pred[:data.num_nodes, :data.num_nodes]
        pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
        num_pos_edges = pos_edge_index.shape[1]
        pred_pos = pred[pos_edge_index[0], pos_edge_index[1]]
        if num_pos_edges > 0:
            neg_mask = torch.ones([num_pos_edges, data.num_nodes], dtype=torch.bool)
            neg_mask[torch.arange(num_pos_edges), pos_edge_index[1]] = False
            pred_neg = pred[pos_edge_index[0]][neg_mask].view(num_pos_edges, -1)
            metrics = self._eval_mrr(pred_pos, pred_neg)
        else:
            metrics = self._eval_mrr(pred_pos, pred_pos)
        return metrics

    @torch.no_grad()
    def evaluator(self, y_hat: torch.Tensor, y: Data) -> torch.Tensor:
        """ Compute Hits@k and Mean Reciprocal Rank (MRR).
        Implementation from LRGB:
        https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/head/inductive_edge.py
        """
        batch = y
        batch_metrics = {
            'hits@1': [],
            'hits@3': [],
            'hits@10': [],
            'mrr': []
        }
        for pred, data in zip(y_hat, batch.to_data_list()):
            metrics = self._eval(pred, data)
            for k, v in metrics.items():
                v = v.mean()
                if v.isnan().item():
                    v = 0.
                batch_metrics[k].append(v)
        for k, v in batch_metrics.items():
            batch_metrics[k] = {
                'metric_sum': sum(v),
                'metric_count': len(v)
            }
        return batch_metrics