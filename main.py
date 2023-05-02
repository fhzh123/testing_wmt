# Import modules
import os
import time
import argparse
# Import custom modules
from training import training
# Utils
from utils import str2bool, path_check, set_random_seed

def main(args):

    # Time setting
    total_start_time = time.time()

    if args.training:
        training(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    user_name = os.getlogin()
    parser = argparse.ArgumentParser(description='Parsing Method')
    # Task setting
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--debuging_mode', action='store_true')
    # Path setting
    parser.add_argument('--preprocess_path', default=f'/HDD/{user_name}/preprocessed', type=str,
                        help='Pre-processed data save path')
    parser.add_argument('--data_path', default='/HDD/dataset/WMT/2016/multi_modal', type=str,
                        help='Original data path')
    parser.add_argument('--model_save_path', default=f'/HDD/{user_name}/model_checkpoint/acl_text_aug', type=str,
                        help='Model checkpoint file path')
    parser.add_argument('--result_path', default=f'/HDD/{user_name}/results/acl_text_aug', type=str,
                        help='Results file path')
    # Training setting
    parser.add_argument('--min_len', default=4, type=int, 
                        help="Sentences's minimum length; Default is 4")
    parser.add_argument('--src_max_len', default=150, type=int, 
                        help="Source sentences's maximum length; Default is 150")
    parser.add_argument('--trg_max_len', default=150, type=int, 
                        help="Target sentences's maximum length; Default is 150")
    parser.add_argument('--num_epochs', default=100, type=int, 
                        help='Training epochs; Default is 100')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Num CPU Workers; Default is 8')
    parser.add_argument('--batch_size', default=16, type=int,    
                        help='Batch size; Default is 16')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Maximum learning rate of warmup scheduler; Default is 5e-5')
    parser.add_argument('--w_decay', default=1e-5, type=float,
                        help="Ralamb's weight decay; Default is 1e-5")
    parser.add_argument('--clip_grad_norm', default=5, type=int, 
                        help='Graddient clipping norm; Default is 5')
    parser.add_argument('--label_smoothing_eps', default=0.05, type=float,
                        help='')
    # Testing setting
    parser.add_argument('--test_batch_size', default=32, type=int, 
                        help='Test batch size; Default is 32')
    parser.add_argument('--beam_size', default=5, type=int, 
                        help='Beam search size; Default is 5')
    parser.add_argument('--beam_alpha', default=0.7, type=float, 
                        help='Beam search length normalization; Default is 0.7')
    parser.add_argument('--repetition_penalty', default=1.3, type=float, 
                        help='Beam search repetition penalty term; Default is 1.3')
    # Seed & Logging setting
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed; Default is 42')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool,
                        help='Using tensorboard; Default is True')
    parser.add_argument('--tensorboard_path', default='./tensorboard_runs', type=str,
                        help='Tensorboard log path; Default is ./tensorboard_runs')
    parser.add_argument('--print_freq', default=100, type=int, 
                        help='Print training process frequency; Default is 100')
    args = parser.parse_args()

    main(args)