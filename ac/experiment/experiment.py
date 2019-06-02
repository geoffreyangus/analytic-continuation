"""
"""
import torch
import torch.nn as nn
import numpy as np

from ac.util import Process


class Experiment(Process):
    """
    """
    def __init__(self, dir, seed=123, cuda=False, devices=[0], model_save_dir="",
                 train_args={}, valid_args={}, targets_dir="", dataset_dir="",
                 dataset_config={}, dataloader_configs=[], model_config={},
                 task_configs=[]):
        """
        """
        super().__init__(dir)

        # set seed for reproducibility
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        # set devices for CUDA compatibility
        self.cuda = cuda
        self.device = devices[0]
        self.devices = devices

        # set location for saved model weights
        self.model_save_dir = model_save_dir
        self.model_save_name = int(time())

        self.train_args = train_args
        self.valid_args = valid_args

        logging.info('Building dataloaders...')
        dataset_class = dataset_config['class']
        dataset_args = dataset_config['args']
        dataset_task_configs = [
            {"task": task_config["task"], **task_config["dataset_args"]}
            for task_config in task_configs
        ]
        dataset_args = {"task_configs": dataset_task_configs, **dataset_args}
        self._build_dataloaders(dataset_class, dataset_args, dataloader_configs)

        logging.info('Building model...')
        model_class = model_config['class']
        model_args = model_config['args']
        model_task_configs = [
            {"task": task_config["task"], **task_config["model_args"]}
            for task_config in task_configs
        ]
        model_args = {"task_configs": model_task_configs, **model_args}
        self._build_model(model_class, model_args)

        # records epoch data in csv
        self.train_history = ExperimentHistory(self.dir)

    def _link_model(self, filepath, symlink):
        """Creates symlink from self.model_save_dir to self.dir"""
        if not os.path.islink(symlink):
            os.symlink(filepath, symlink)

    def is_trained(self):
        """Returns true if the model has been trained for at least one epoch."""
        return os.path.isdir(os.path.join(self.dir, "last"))


    def _build_dataloaders(self, dataset_class, dataset_args,
                           dataset_task_configs, dataloader_configs):
        """
        """
        self.datasets = {}
        self.dataloaders = {}
        for dataloader_config in dataloader_configs:
            split = dataloader_config["split"]
            dataloader_class = dataloader_config["dataloader_class"]
            dataloader_args = dataloader_config["dataloader_args"]
            logging.info(f"Loading {split} data")
            self._build_dataloader(split, dataset_class, dataset_args,
                                   dataloader_class, dataloader_args)

    def _build_dataloader(self, split, dataset_class, dataset_args,
                          dataloader_class, dataloader_args):
        """
        """
        dataset = getattr(datasets, dataset_class)(split=split, **dataset_args)
        self.datasets[split] = dataset

        dataloader = (getattr(dataloaders, dataloader_class)(dataset, **dataloader_args))
        self.dataloaders[split] = dataloader

    def _build_model(self, model_class, model_args):
        """
        Builds the model.

        If the model was previously trained, it is loaded from a previous model.
        """
        model_class = getattr(models, model_class)
        self.model = model_class(task_configs, cuda=self.cuda, devices=self.devices,
                                 **model_args)

    def _run(self, overwrite=False, mode="train", train_split="train", eval_split="valid"):
        """
        """
        if mode == "train":
            self.train(train_split=train_split, valid_split=eval_split,
                       overwrite=overwrite)
        elif mode == "eval":
            self.evaluate(eval_split=eval_split)

    def evaluate(self, eval_split="valid"):
        """
        """
        metrics = self.model.score(self.dataloaders[eval_split],
                                   **self.evaluate_args)
        self._save_metrics(self.model_dir, metrics.metrics,
                           f"eval_{eval_split}")
        return metrics

    def train(self, train_split="train", valid_split="valid", overwrite=False):
        """
        """
        assert not self.is_trained() or overwrite, (
            "The model has already been trained."
        )

        best_score = None
        for epoch_num, train_metrics in enumerate(self.model.train_model(
            dataloader=self.dataloaders[train_split], **self.train_args)):

            val_metrics = self.model.score(self.dataloaders[valid_split],
                                           **self.evaluate_args)

            self._save_weights(name="last")
            self._save_epoch(epoch_num, train_metrics,
                             val_metrics, name="last")

            metrics = {"train": train_metrics.metrics,
                       "valid": val_metrics.metrics}
            self.train_history.record_epoch(metrics, self.model.scheduler.get_lr()[0])
            self.train_history.write()

            if (best_score is None or val_metrics.primary_metric > best_score):
                self._save_weights(name="best")
                self._save_epoch(epoch_num, train_metrics,
                                 val_metrics, name="best")
                best_score = val_metrics.primary_metric
        return metrics

    def get_history(self):
        """
        """
        return ExperimentHistory(self.dir)

    def _save_weights(self, name="last"):
        """
        """
        remote_weights_path = os.path.join(self.model_save_dir,
                                           f"{self.experiment_t}_{name}_weights")
        self.model.save_weights(remote_weights_path)

        # create a symbolic link to model weights
        self.model_dir = os.path.join(self.dir, name)
        ensure_dir_exists(self.model_dir)
        link_weights_path = os.path.join(self.model_dir, "weights.link")
        self._link_model(remote_weights_path, link_weights_path)

    def _save_epoch(self, epoch_num, train_metrics, valid_metrics, name="last"):
        """
        """
        save_dir = os.path.join(self.dir, name)
        ensure_dir_exists(save_dir)
        logging.info("Saving checkpoint...")

        # records the most recent training epoch
        self._save_metrics(save_dir, train_metrics, "train")
        self._save_metrics(save_dir, valid_metrics, "valid")

    def _save_metrics(self, metrics_dir, metrics, split):
        """
        """
        save_dict_to_json(os.path.join(metrics_dir, f"{split}_metrics.json"),
                          metrics.metrics)
        preds_df = metrics.get_preds()
        preds_df.to_csv(os.path.join(metrics_dir, f"{split}_preds.csv"))
