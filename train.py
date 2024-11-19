# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from config import cfg
import torch
from base import Trainer
import torch.backends.cudnn as cudnn

from base import Tester
from tqdm import tqdm
import numpy as np
import time

#from torch.utils.tensorboard import SummaryWriter
import wandb
'''
command for opening tensorboard is : 
tensorboard --logdir=your/root/file/path/output/tensorboard_log
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', dest='gpu_ids')
    parser.add_argument('--continue', default=False, dest='continue_train', action='store_true')
    parser.add_argument('--wandb', default=False, dest='use_wandb', action='store_true')
    args = parser.parse_args()


    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train, args.use_wandb)
    cudnn.benchmark = True
    if cfg.use_wandb:
        wandb.init(
                        project = "A2J",
                        id = "depth1layer_mamba_backboneS_509095_RHP_full1",
                        resume = True
                    )
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
    
    #tbwriter = SummaryWriter(cfg.tensorboard_dir)

    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        loss_100_total = 0
        loss_100_anchor = 0
        loss_100_reg = 0
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            loss = trainer.model(inputs, targets, meta_info, 'train')
            loss = {k:loss[k].mean() for k in loss}
            loss_100_total = loss_100_total + loss['total_loss']
            loss_100_anchor = loss_100_anchor + loss['Cls_loss']
            loss_100_reg = loss_100_reg + loss['Reg_loss']
            # backward
            loss['total_loss'].backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            if itr % 100 == 0:
                screen = [
                    'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                    'lr: %g' % (trainer.get_lr()),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (
                        trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                    ]
                screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
                trainer.logger.info(' '.join(screen))
            if itr !=0 and itr % 100 == 0: 
                if cfg.use_wandb:
                    wandb.log({'step/loss': loss_100_total / 100})
                    wandb.log({'step/loss/anchor_loss': loss_100_anchor/100})
                    wandb.log({'step/loss/reg_loss': loss_100_reg/100})
                #print(f"total_loss is {loss_100_total / 100}")
                loss_100_total = 0
                loss_100_anchor = 0
                loss_100_reg = 0
            # if itr % 100 ==0: 
            #     tbwriter.add_scalar('loss/total_loss', loss['total_loss'], epoch*len(trainer.batch_generator)+itr)
                
            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
        
        # save model
        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)
        #mpjpe_dict = test_per_epoch(epoch)
        #wandb.log({'test/mpjpe': mpjpe_dict['total'], "epoch": epoch})
    #     mpjpe_dict = test_per_epoch(epoch)
    #     if cfg.use_single_hand_dataset:
    #         tbwriter.add_scalar('mpjpe/single_hand_total', mpjpe_dict['single_hand_total'], epoch)
    #         tbwriter.add_scalar('mpjpe/single_hand_2d', mpjpe_dict['single_hand_2d'], epoch)
    #         tbwriter.add_scalar('mpjpe/single_hand_depth', mpjpe_dict['single_hand_depth'], epoch)
    #     if cfg.use_inter_hand_dataset:
    #         tbwriter.add_scalar('mpjpe/inter_hand_total', mpjpe_dict['inter_hand_total'], epoch)
    #         tbwriter.add_scalar('mpjpe/inter_hand_2d', mpjpe_dict['inter_hand_2d'], epoch)
    #         tbwriter.add_scalar('mpjpe/inter_hand_depth', mpjpe_dict['inter_hand_depth'], epoch)
    #     if cfg.use_single_hand_dataset and cfg.use_inter_hand_dataset:
    #         tbwriter.add_scalar('mpjpe/total', mpjpe_dict['total'], epoch)
    #     if hand_accuracy is not None:
    #         tbwriter.add_scalar('hand_accuracy', hand_accuracy, epoch)
    #     if mrrpe is not None:
    #         tbwriter.add_scalar('mrrpe', mrrpe, epoch)
    # tbwriter.close()
    



def test_per_epoch(test_epoch):

    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    args.test_set = 'test'
    args.test_epoch = str(test_epoch)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator(args.test_set)
    tester._make_model()

    preds = {'joint_coord': [], 'inv_trans': [], 'joint_valid': [] }

    with torch.no_grad():
       for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
            
            # forward
            out = tester.model(inputs, targets, meta_info, 'test')

            joint_coord_out = out['joint_coord'].cpu().numpy()
            inv_trans = out['inv_trans'].cpu().numpy()
            joint_vaild = out['joint_valid'].cpu().numpy()


            preds['joint_coord'].append(joint_coord_out)
            preds['inv_trans'].append(inv_trans)
            preds['joint_valid'].append(joint_vaild)

            
    # evaluate
    preds = {k: np.concatenate(v) for k,v in preds.items()}
    mpjpe_dict = tester._evaluate(preds)   
    return mpjpe_dict



if __name__ == "__main__":
    main()
