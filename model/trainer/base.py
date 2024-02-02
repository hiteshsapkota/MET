import abc 
import torch 
import os.path as osp 
from model.logger import Logger 
from model.utils import (Averager, Timer)

class Trainer(object, metaclass = abc.ABCMeta):
    def __init__(self, args):
        self.args = args 

        self.logger = Logger(args, osp.join(args.save_path))

        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch*args.max_epoch

        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics 
        self.trlog = {}
        self.trlog['max_acc'] = 0.0
        self.trlog['max_acc_epoch'] = 0.0
        self.trlog['max_acc_interval'] = 0.0
        

    
    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self, data_loader):
        pass
    
    @abc.abstractmethod
    def evaluate_test(self, data_loader):
        pass    
    
    @abc.abstractmethod
    def final_record(self):
        pass 


    def try_evaluate(self, epoch):
        args = self.args
        if self.train_epoch%args.eval_interval==0:
            vl, vaccm, vaccs, vaucmp, vaucsp, vaucmd, vaucsd, vaucms, vaucss, vaucmedl, vaucsedl = self.evaluate(self.val_loader)  
            self.logger.add_scalar('val_loss', float(vl), self.train_epoch)
            self.logger.add_scalar('val_acc', float(vaccm),  self.train_epoch)
            self.logger.add_scalar('val_auc_prob', float(vaucmp), self.train_epoch)
            self.logger.add_scalar('val_auc_dist', float(vaucmd), self.train_epoch)
            self.logger.add_scalar('val_auc_snatcher', float(vaucms), self.train_epoch)
            self.logger.add_scalar('val_auc_edl', float(vaucmedl), self.train_epoch)
            
            print('epoch {}, val, loss={:.4f} acc={:.4f}+{:.4f}, auc: prob: {:.4f}+{:.4f}, dist: {:.4f}+{:.4f}, snatcher: {:.4f}+{:.4f}, edl: {:.4f}+{:.4f}'.format(epoch, vl, vaccm, vaccs, vaucmp, vaucsp, vaucmd, vaucsd, vaucms, vaucss, vaucmedl, vaucsedl))

            if vaccm>=self.trlog['max_acc']:
                self.trlog['max_acc'] = vaccm
                self.trlog['max_acc_interval'] = vaccs
                self.trlog['max_acc_epoch'] = self.train_epoch
                self.save_model('max_acc')

    
    def try_logging(self, tl1, tl2, ta, tg = None):
        args = self.args 
        if self.train_step%args.log_interval==0:
            print('epoch {}, train {:06g}/{:06g}, total loss={:.4f}, loss={:.4f} acc={:.4f}, lr={:.4g}'
                  .format(self.train_epoch,
                          self.train_step,
                          self.max_steps,
                          tl1.item(), tl2.item(), ta.item(),
                          self.optimizer.param_groups[0]['lr']))
            self.logger.add_scalar('train_total_loss', tl1.item(), self.train_step)
            self.logger.add_scalar('train_loss', tl2.item(), self.train_step)
            self.logger.add_scalar('train_acc',  ta.item(), self.train_step)
            if tg is not None:
                self.logger.add_scalar('grad_norm',  tg.item(), self.train_step)
            print('data_timer: {:.2f} sec, '     \
                  'forward_timer: {:.2f} sec,'   \
                  'backward_timer: {:.2f} sec, ' \
                  'optim_timer: {:.2f} sec'.format(
                        self.dt.item(), self.ft.item(),
                        self.bt.item(), self.ot.item()))
            
            self.logger.dump()


    
    def save_model(self, name):
        if self.args.open_loss:
            save_path =  osp.join(self.args.save_path, str(self.args.shot)+'_s_'+str(self.args.run)+'_r_'+str(self.args.balance)+'_bal_'+str(self.args.open_loss_coeff)+'_olf_'+self.args.loss_type+'_'+name +'.pth')
        else:
            save_path = osp.join(self.args.save_path, str(self.args.shot)+'_s_'+str(self.args.run)+'_r_'+str(self.args.balance)+'_bal_'+self.args.loss_type+'_'+name +'.pth')

        torch.save(
            dict(params=self.model.state_dict()),save_path
            
        )
    
    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )













