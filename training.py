import os
import math
import logging
from tqdm import tqdm
from time import time

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR

from transformers import AutoTokenizer

from dataset import CustomDataset
from model import TransformerModel
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

def training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start training!')

    #===================================#
    #=============Data Load=============#
    #===================================#

    write_log(logger, "Load data...")

    start_time = time()

    src_list = dict()
    trg_list = dict()

    # 1) Train data load
    with open(os.path.join(args.data_path, 'train.de'), 'r') as f:
        src_list['train'] = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(args.data_path, 'train.en'), 'r') as f:
        trg_list['train'] = [x.replace('\n', '') for x in f.readlines()]

    # 2) Valid data load
    with open(os.path.join(args.data_path, 'val.de'), 'r') as f:
        src_list['valid'] = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(args.data_path, 'val.en'), 'r') as f:
        trg_list['valid'] = [x.replace('\n', '') for x in f.readlines()]

    # 3) Test data load
    with open(os.path.join(args.data_path, 'test.de'), 'r') as f:
        src_list['test'] = [x.replace('\n', '') for x in f.readlines()]
    with open(os.path.join(args.data_path, 'test.en'), 'r') as f:
        trg_list['test'] = [x.replace('\n', '') for x in f.readlines()]

    write_log(logger, 'Data loading done!')

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')
    model = TransformerModel()
    model.to(device)

    # 2) Dataloader setting
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    dataset_dict = {
        'train': CustomDataset(tokenizer=tokenizer,
                               src_list=src_list['train'], trg_list=trg_list['train'],
                               src_max_len=args.src_max_len, trg_max_len=args.trg_max_len),
        'valid': CustomDataset(tokenizer=tokenizer,
                               src_list=src_list['valid'], trg_list=trg_list['valid'],
                               src_max_len=args.src_max_len, trg_max_len=args.trg_max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=False,
                            batch_size=args.batch_size, shuffle=True,
                            pin_memory=True, num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    # 3) Optimizer & scheduler setting
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-5
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.lr, eps=1e-8, weight_decay=args.w_decay)
    scheduler = WarmupLinearSchedule(optimizer,
                                    warmup_steps=int(len(dataloader_dict['train'])*2),
                                    t_total=len(dataloader_dict['train'])*args.num_epochs)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps, ignore_index=model.pad_idx).to(device)

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        write_log(logger, 'Resume model...')
        save_file_name = os.path.join(args.model_save_path, args.data_name, args.encoder_model_type, 'checkpoint_test.pth.tar')
        checkpoint = torch.load(save_file_name)
        start_epoch = checkpoint['epoch'] - 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'Traing start!')
    best_val_loss = 1e+4

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        start_time_e = time()

        write_log(logger, 'Training start...')

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                write_log(logger, 'Validation start...')
                val_loss = 0
                val_acc = 0
                model.eval()

            for i, batch_iter in enumerate(tqdm(dataloader_dict[phase], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

                # Optimizer setting
                optimizer.zero_grad(set_to_none=True)

                # Input, output setting
                src_sequence = batch_iter[0]
                src_att = batch_iter[1]
                src_sequence = src_sequence.to(device, non_blocking=True)
                src_att = src_att.to(device, non_blocking=True)

                trg_sequence = batch_iter[2]
                trg_att = batch_iter[3]
                trg_sequence = trg_sequence.to(device, non_blocking=True)
                trg_att = trg_att.to(device, non_blocking=True)

                # Output pre-processing
                trg_sequence_gold = trg_sequence[:, 1:]
                non_pad = trg_sequence_gold != model.pad_idx
                trg_sequence_gold = trg_sequence_gold[non_pad].contiguous().view(-1)

                # Train
                if phase == 'train':
                    predicted = model(src_input_ids=src_sequence, src_attention_mask=src_att,
                                      trg_input_ids=trg_sequence[:, :-1], trg_attention_mask=trg_att[:, :-1], 
                                      non_pad_position=non_pad)
                    loss = F.cross_entropy(predicted, trg_sequence_gold, ignore_index=model.pad_idx)
                    loss.backward()
                    clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    scheduler.step()

                    # Print loss value only training
                    if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                        # Non padding masking
                        non_pad = trg_sequence != model.pad_idx
                        acc = (predicted.max(dim=1)[1] == trg_sequence_gold).sum() / len(trg_sequence_gold)
                        iter_log = "[Epoch:%03d][%03d/%03d] train_seq_loss:%03.2f | train_acc:%03.2f%% | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                            (epoch, i, len(dataloader_dict['train']), 
                            loss.item(), acc*100, optimizer.param_groups[0]['lr'], 
                            (time() - start_time_e) / 60)
                        write_log(logger, iter_log)
                        freq = 0
                    freq += 1

                # Validation
                if phase == 'valid':
                    with torch.no_grad():
                        predicted = model(src_input_ids=src_sequence, src_attention_mask=src_att,
                                        trg_input_ids=trg_sequence[:, :-1], trg_attention_mask=trg_att[:, :-1], 
                                        non_pad_position=non_pad)
                        loss = F.cross_entropy(predicted, trg_sequence_gold, ignore_index=model.pad_idx)
                    val_loss += loss.item()
                    # Non padding masking
                    non_pad = trg_sequence != model.pad_idx
                    val_acc += (predicted.max(dim=1)[1] == trg_sequence_gold).sum() / len(trg_sequence_gold)

        val_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Validation CrossEntropy Loss: %3.3f' % val_loss)
        write_log(logger, 'Validation Accuracy: %3.3f' % val_acc)

        save_file_name = os.path.join(args.model_save_path, f'checkpoint.pth.tar')
        if val_loss < best_val_loss:
            write_log(logger, 'Model checkpoint saving...')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, save_file_name)
            best_val_loss = val_loss
            best_aug_epoch = epoch
        else:
            else_log = f'Still {best_aug_epoch} epoch Loss({round(best_val_loss, 2)}) is better...'
            write_log(logger, else_log)