"""
"""
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import sklearn.metrics as skl

from ac.util import place_on_cpu, get_batch_size, array_like_concat


class Metrics:
    """
    """
    def __init__(self, metric_configs=[]):
        """
        Args:
            metrics_configs (list, dict)   list of function names to use for eval
        """
        self.metric_configs = metric_configs

        self.metrics = defaultdict(dict)
        self.global_metrics = defaultdict(int)
        self.precomputed_keys = None

        self.preds = defaultdict(list)
        self.targets = defaultdict(list)
        self.info = []

        self.total_size = 0

    def get_metric(self, metric, task=None):
        """
        """
        if task is None:
            return self.global_metrics[metric]
        else:
            return self.metrics[task][metric]

    def get_preds(self):
        """
        """
        index = [info["sequence_id"]for info in self.info]
        task2df = {}
        for task in self.preds.keys():
            preds = []
            targets = []
            for batch_probs, batch_targets in zip(self.preds[task], self.targets[task]):
                targets.extend(batch_targets)
                preds.extend(batch_probs)
            task_df = pd.DataFrame(
                data={
                    "target": [str(target) for target in targets],
                    "pred": [str(pred) for pred in preds]
                },
                index=index
            )
            task2df[task] = task_df

        return pd.concat(task2df.values(), axis=1, keys=task2df.keys())

    def add(self, preds, targets, info, precomputed_metrics={}):
        tasks = list(preds.keys())
        batch_size = get_batch_size(list(targets.values())[0])
        for task in tasks:
            task_targets = targets[task]
            task_preds = preds[task]
            if(get_batch_size(task_targets) != get_batch_size(task_preds)):
                raise ValueError("preds must match targets in first dim.")

            self.preds[task].append(place_on_cpu(task_preds))
            self.targets[task].append(place_on_cpu(task_targets))
        self.info.extend(info)

        # include precomputed keys in global metrics
        if self.precomputed_keys != None and \
           self.precomputed_keys != set(precomputed_metrics.keys()):
            raise ValueError("must always supply same precomputed metrics.")
        elif self.precomputed_keys == None:
            self.precomputed_keys = set(precomputed_metrics.keys())

        for key, value in precomputed_metrics.items():
            self.global_metrics[key] = ((self.total_size * self.global_metrics[key] +
                                              batch_size * value) /
                                             (batch_size + self.total_size))

        self.total_size += batch_size

    def compute(self):
        """
        Computes metrics on the collected set of preds and targets
        """
        # call all metric_fns, detach since output has require grad
        for metric_config in self.metric_configs:
            self._compute_metric(**metric_config)

    def _compute_metric(self, fn, args={}, name=None, tasks=None,
                        is_primary=False, primary_task='primary',
                        compute_task_mean=True):
        """
        Computes a given metric.

        Args:
            fn (string) a string corresponding to the metric function to apply.
            args (dict) arguments associated with the given metric function.
            name (string) (optional) human-readable format for function.
            tasks (list) (optional) desired tasks for metrics. If None, metrics
                will be computed for all tasks.
            is_primary (boolean) determines the metric used for checkpointing
                best performing models.
        """
        name = name if name is not None else fn

        values = []
        for task in self.preds.keys():
            if tasks is not None and task not in tasks:
                continue

            all_preds = []
            all_targets = []
            for batch_preds, batch_targets in zip(self.preds[task], self.targets[task]):
                if type(batch_preds) is torch.Tensor:
                    # flatten dimensions
                    batch_preds = batch_preds.view(-1, batch_preds.shape[-1]).squeeze(-1)
                    batch_targets = batch_targets.view(-1).squeeze(-1)
                all_preds.append(batch_preds)
                all_targets.append(batch_targets)

            all_preds = array_like_concat(all_preds)
            all_targets = array_like_concat(all_targets)

            value = globals()[fn](all_preds, all_targets, **args)
            self.metrics[task][name] = value
            values.append(value)

            if is_primary and task == primary_task:
                self.primary_metric = self.metrics[task][name]

        if compute_task_mean:
            self.metrics['_average'][name] = np.mean(values)
        if '_average' == primary_task:
            self.primary_metric = self.metrics['_average'][name]

        self.global_metrics[name] = np.mean(values)


def mean_squared_error(preds, targets):
    """
    """
    return skl.mean_squared_error(targets, preds)
