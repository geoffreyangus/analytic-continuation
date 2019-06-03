"""
Utility functions.
"""
from jsmin import jsmin
import json
import os
import logging
import smtplib
import traceback
import socket
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import collections

import torch
import numpy as np

class Process():
    """
    """

    def __init__(self, dir, name=None):
        self.dir = dir
        set_logger(os.path.join(dir, "process.log"),
                   level=logging.INFO, console=True)

        params = load_params(self.dir)
        logging.info(json.dumps(params, indent=2))

        process_class = params["process_class"]
        log_title(process_class)

    @classmethod
    def load_from_dir(cls, process_dir, name=None):
        params = load_params(process_dir)
        process = cls(process_dir, **params["process_args"])
        return process

    def _run(self, overwrite=False):
        """
        """
        pass

    def run(self, overwrite=False, notify=False, commit=False, **kwargs):
        """
        """
        if os.path.isfile(os.path.join(self.dir, 'stats.csv')) and not overwrite:
            print("Process already run.")
            return False

        if notify:
            try:
                self._run(overwrite)
            except:
                tb = traceback.format_exc()
                self.notify_user(error=tb)
                return False
            else:
                self.notify_user()
                return True
        self._run(overwrite, **kwargs)
        return True

    def notify_user(self, error=None):
        """
        Notify the user by email if there is an exception during the process.
        If the program completes without error, the user will also be notified.
        """
        # read params
        params = load_params(self.dir)
        params_string = str(params)

        if error is None:
            subject = "Process Completed: " + self.dir
            message = "Hello!"
        else:
            subject = "Process Error: " + self.dir
            message = "Hello!"
            logging.error(error)

        message = "\n".join(message)
        send_email(subject, message, to_addr='skivelso@stanford.edu')
        send_email(subject, message, to_addr='gangus@stanford.edu')


def load_params(process_dir):
    """
    Loads the params file in the process directory specified by process_dir.
    @param process_dir (str)
    @returns params (dict)
    """
    if os.path.exists(os.path.join(process_dir, "params.json")):
        params_path = os.path.join(process_dir, "params.json")
        with open(params_path) as f:
            params = json.load(f)

    elif os.path.exists(os.path.join(process_dir, "params.jsonc")):
        params_path = os.path.join(process_dir, "params.jsonc")
        with open(params_path) as f:
            minified_json = jsmin(f.read())
            params = json.loads(minified_json)

    else:
        raise Exception(f"No params.json file found at {[process_dir]}.")

    return params


def send_email(subject, message, to_addr):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("analyticcontinuation2019", "experiments")

    msg = MIMEMultipart()       # create a message

    # add in the actual person name to the message template

    # setup the parameters of the message
    msg['From'] = "analyticcontinuation2019@gmail.com"
    msg['To'] = to_addr
    msg['Subject'] = subject

    # add in the message body
    msg.attach(MIMEText(message, 'plain'))

    problems = server.sendmail("analyticcontinuation2019@gmail.com",
                               to_addr,
                               msg.as_string())
    server.quit()


def set_logger(log_path, level=logging.INFO, console=True):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logging.basicConfig(format='')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    if console and False:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def log_title(title):
    """
    """
    logging.info("{}".format(title))
    logging.info("Sophia Kivelson and Geoff Angus")
    logging.info("Stanford University – 2019")
    logging.info("---------------------------------")


def ensure_dir_exists(dir):
    """
    Ensures that a directory exists. Creates it if it does not.
    args:
        dir     (str)   directory to be created
    """
    if not(os.path.exists(dir)):
        ensure_dir_exists(os.path.dirname(dir))
        os.mkdir(dir)


def extract_kwargs(kwargs_tuple):
    """  Converts  a tuple of kwarg tokens to a dictionary. Expects in format
    (--key1, value1, --key2, value2)
    Args:
        kwargs_tuple (tuple(str)) tuple of list
    """
    if len(kwargs_tuple) == 0:
        return {}
    assert kwargs_tuple[0][:
                           2] == "--", f"No key for first kwarg {kwargs_tuple[0]}"
    curr_key = None
    kwargs_dict = {}
    for token in kwargs_tuple:
        if token[:2] == "--":
            curr_key = token[2:]
        else:
            kwargs_dict[curr_key] = token

    return kwargs_dict


def array_like_concat(items):
    """
    Concatenates array-like `items` along dimension 0.
    """
    if len(items) < 1:
        raise ValueError("items is empty")

    if len(set([type(item) for item in items])) != 1:
        raise TypeError("items are not of the same type")

    if isinstance(items[0], list):
        return sum(items, [])
    elif isinstance(items[0], torch.Tensor):
        # zero-dimensional tensors cannot be concatenated
        if not items[0].shape:
            items = [item.expand(1) for item in items]
        return torch.cat(items, dim=0)
    else:
        raise TypeError(f"Unrecognized type f{type(items[0])}")


def array_like_stack(items):
    """
    """
    if len(items) < 1:
        raise ValueError("items is empty")

    if len(set([type(item) for item in items])) != 1:
        raise TypeError("items are not of the same type")

    if isinstance(items[0], list):
        return items
    elif isinstance(items[0], torch.Tensor):
        return torch.stack(items, dim=0)
    elif isinstance(items[0], np.ndarray):
        return np.stack(items, axis=0)
    else:
        raise TypeError(f"Unrecognized type f{type(items[0])}")


def get_batch_size(data):
    """
    """
    if isinstance(data, list):
        return len(data)
    elif isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
        return data.shape[0]
    else:
        raise TypeError(f"Unrecognized type f{type(data)}")


def save_dict_to_json(json_path, d):
    """
    Saves a python dictionary into a json file
    Args:
        d           (dict) of float-castable values (np.float, int, float, etc.)
        json_path   (string) path to json file
    """
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=4, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    """
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def flatten_nested_dicts(d, parent_key='', sep='_'):
    """
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_nested_dicts(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def place_on_gpu(data, device=0):
    """
    Recursively places all 'torch.Tensor's in data on gpu and detaches.
    If elements are lists or tuples, recurses on the elements. Otherwise it
    ignores it.
    source: inspired by place_on_gpu from Snorkel Metal
    https://github.com/HazyResearch/metal/blob/master/metal/utils.py
    """
    data_type = type(data)
    if data_type in (list, tuple):
        data = [place_on_gpu(data[i], device) for i in range(len(data))]
        data = data_type(data)
        return data
    elif data_type is dict:
        data = {key: place_on_gpu(val, device) for key, val in data.items()}
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def place_on_cpu(data):
    """
    Recursively places all 'torch.Tensor's in data on cpu and detaches from computation
    graph. If elements are lists or tuples, recurses on the elements. Otherwise it
    ignores it.
    source: inspired by place_on_gpu from Snorkel Metal
    https://github.com/HazyResearch/metal/blob/master/metal/utils.py
    """
    data_type = type(data)
    if data_type in (list, tuple):
        data = [place_on_cpu(data[i]) for i in range(len(data))]
        data = data_type(data)
        return data
    elif data_type is dict:
        data = {key: place_on_cpu(val) for key, val in data.items()}
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().detach()
    else:
        return data
