import numpy as np 
import torch 
import argparse
import random 
from model.utils import (set_gpu)
from model.trainer.fsl_trainer import FSLTrainer

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--num_eval_episodes', type=int, default=600)
    parser.add_argument('--model_class', type=str, default='FEAT', 
                        choices=['BILSTM', 'DeepSet', 'GCN', 'FEAT'])
    parser.add_argument('--use_euclidean', type=bool, default=True)
    parser.add_argument('--backbone_class', type=str, default='Res12',
                        choices=['ConvNet', 'Res12', 'Res18'])
    
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'Cifar100', 'Caltech'])
    
    parser.add_argument('--closed_way', type=int, default=5)
    parser.add_argument('--open_way', type=int, default=5)
    parser.add_argument('--closed_eval_way', type=int, default=5)

    parser.add_argument('--open_eval_way', type=int, default=5)

    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--eval_shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--balance', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=64)
    parser.add_argument('--temperature2', type=float, default=64)  # the temperature in the  

    # optimization parameters
    parser.add_argument('--orig_imsize', type=int, default=-1) # -1 for no cache, and -2 for no resize, only for MiniImageNet and CUB
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lr_mul', type=float, default=10) 
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.5)    
    parser.add_argument('--fix_BN', action='store_true', default=False)     # means we do not update the running mean/var in BN, not to freeze BN
    parser.add_argument('--augment',   action='store_true', default=False)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--init_weights', type=str, default='./saves/initialization/miniimagenet/Res12-pre.pth', choices=['./saves/initialization/miniimagenet/Res12-pre.pth', './saves/initialization/tierdimagenet/Res12-pre.pth', './saves/initialization/cifar100/Res12-pre.pth', './saves/initialization/caltech/Res12-pre.pth'])

    # usually untouched parameters
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005) # we find this weight decay value works the best
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--annealing_step', type=float, default=10)   
    parser.add_argument('--loss_type_no', type = int, default = 1)
    parser.add_argument('--loss_type', type = str, default = 'edl_loss') 
    parser.add_argument('--edl_loss', type = str, default = 'mse')
    parser.add_argument('--open_loss', type = bool, default = False)
     parser.add_argument('--open_loss_coeff', type = float, default = 1.0)

    args = parser.parse_args()
    args.eval_shot = args.shot
    if args.loss_type_no==0:
        args.loss_type='edl_loss'
    else:
        args.loss_type='ce_loss'

    if args.dataset=='MiniImageNet':
        args.data_path = './data/miniimagenet'
    elif args.dataset =='Tieredimagenet':
        args.data_path = './data/tieredimagenet'
    elif args.dataset == 'Cifar100':
        args.data_path = './data/cifar100'
    elif args.dataset =='Caltech':
        args.data_path = 'Caltech'
   
    
    if args.data_path is None:
        raise ValueError ('Specify your data path')
    
    args.eval_way = args.closed_way+args.open_eval_way
    
 
    args.save_path = './checkpoints'
    prefix = 'mini' if args.dataset=='MiniImageNet' else 'tiered'
    args.weight_name = '%s-feat-%d-shot.pth'%(prefix, args.shot)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    set_gpu(args.gpu)

    trainer = FSLTrainer(args)
   trainer.train()
    trainer.evaluate_test()
    trainer.final_record()
    print(args.save_path)


    

    

















    
            
