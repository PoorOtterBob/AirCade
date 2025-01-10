import argparse
import sys

def get_tkde_config():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--num', type=int, default='')         # the number of datasets in Knowair
    parser.add_argument('--seed', type=int, default=3028)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=13)    # more information for training for 5
    parser.add_argument('--output_dim', type=int, default=1)

    # for data segment
    parser.add_argument('--max_increase_ratio', type=float, default=0) # dataset-driven hyperpara.

    # for training
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--test_increase_ratio', type=float, default=0) # must less than max_changable_ratio
    parser.add_argument('--test_decrease_ratio', type=float, default=0) # must less than 1
    parser.add_argument('--sood', type=int, default=0) # if spatial shift or not
    parser.add_argument('--tood', type=int, default=0) # if temporal shift or not
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)

    # For Frechet embbeding
    parser.add_argument('--c', type=int, default=1) 
    parser.add_argument('--adj_type', type=str, default='origin')
    parser.add_argument('--spatial_noise', type=bool, default=True)

    # For EERM mask
    parser.add_argument('--mask_patience', type=int, default=5)
    parser.add_argument('--K_t', type=int, default=4) # num of t_mask
    parser.add_argument('--K_s', type=int, default=4) # num of t_mask
    parser.add_argument('--t_sample_ratio', type=float, default=0.1) # ratio of mask
    parser.add_argument('--s_sample_ratio', type=float, default=0.2) # ratio of mask
    return parser

def get_public_config():
    parser = argparse.ArgumentParser()   
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--num', type=int, default='') # the number of datasets in Knowair
    parser.add_argument('--seed', type=int, default=3028)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=13)    # more information for training for 5
    parser.add_argument('--output_dim', type=int, default = 1)      
    parser.add_argument('--c', type=int, default='1')  
    # for data segment
    parser.add_argument('--max_increase_ratio', type=float, default=0) # dataset-driven hyperpara.

    # for training
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--test_increase_ratio', type=float, default=0) # must less than max_changable_ratio
    parser.add_argument('--test_decrease_ratio', type=float, default=0) # must less than 1
    parser.add_argument('--sood', type=int, default=0) # if spatial shift or not
    parser.add_argument('--tood', type=int, default=0) # if temporal shift or not
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    
    return parser

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')