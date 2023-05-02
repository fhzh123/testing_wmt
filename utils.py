# Import modules
import os
import sys
import time
import tqdm
import logging
import argparse
# Import PyTorch
import torch.nn.functional as F

def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

def path_check(args):
    # Preprocessing Path Checking
    if not os.path.exists(args.preprocess_path):
        os.mkdir(args.preprocess_path)

    if not os.path.exists(os.path.join(args.preprocess_path, args.tokenizer)):
        os.mkdir(os.path.join(args.preprocess_path, args.tokenizer))

    # Model Checkpoint Path Checking
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    if not os.path.exists(os.path.join(args.model_save_path, args.tokenizer)):
        os.mkdir(os.path.join(args.model_save_path, args.tokenizer))

    # Testing Results Path Checking
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.stream = sys.stdout

    def flush(self):
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, "flush"):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            raise
        except Exception:
            self.handleError(record)

def write_log(logger, message):
    if logger:
        logger.info(message)

def get_tb_exp_name(args:argparse.Namespace):
    """
    Get the experiment name for tensorboard experiment.
    """

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.localtime())

    exp_name = str()
    exp_name += "%s - " % args.model_type

    if args.training:
        exp_name += 'TRAIN - '
        exp_name += "BS=%i_" % args.batch_size 
        exp_name += "EP=%i_" % args.num_epochs
        exp_name += "LR=%.4f_" % args.lr
    elif args.testing:
        exp_name += 'TEST - '
        exp_name += "BS=%i_" % args.batch_size
    exp_name += "TS=%s" % ts

    return exp_name

def set_random_seed(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)