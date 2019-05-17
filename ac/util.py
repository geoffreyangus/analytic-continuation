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
    server.login("analytic-continuation", "experiments")

    msg = MIMEMultipart()       # create a message

    # add in the actual person name to the message template

    # setup the parameters of the message
    msg['From'] = "analytic-continuation@gmail.com"
    msg['To'] = to_addr
    msg['Subject'] = subject

    # add in the message body
    msg.attach(MIMEText(message, 'plain'))

    problems = server.sendmail("analytic-continuation@gmail.com",
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


def send_email(subject, message, to_addr):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("analytic-continuation", "experiments")

    msg = MIMEMultipart()       # create a message

    # add in the actual person name to the message template

    # setup the parameters of the message
    msg['From'] = "analytic-continuation@gmail.com"
    msg['To'] = to_addr
    msg['Subject'] = subject

    # add in the message body
    msg.attach(MIMEText(message, 'plain'))

    problems = server.sendmail("analytic-continuation@gmail.com",
                               to_addr,
                               msg.as_string())
    server.quit()


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
