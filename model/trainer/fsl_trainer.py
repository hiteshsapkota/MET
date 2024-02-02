
from model.trainer.base import Trainer 
from model.trainer.helpers import (get_dataloader, prepare_model, prepare_optimizer)
from edl_losses import select_edl_loss, relu_evidence
from model.utils import (pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval)

from sklearn.metrics import roc_auc_score, roc_curve
import torch 
import time 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import numpy as np 
import os.path as osp 
from tqdm import tqdm
from torch import nn 
import libmr

cos = nn.CosineSimilarity()
from torch.autograd import Variable
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse = False)

def get_onehot_encoder(y):
    le = label_encoder.fit(y)
    integer_encoded = le.transform(y).reshape(-1, 1)
    y_hot = onehot_encoder.fit_transform(integer_encoded)
    return y_hot

def calc_auroc(known_scores, unknown_scores):
    y_true = np.array([0] * len(known_scores) + [1] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    
    return auc_score

    



class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)


    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.closed_way, dtype=torch.int16).repeat(args.query)
        label_hot = get_onehot_encoder(label.data.cpu().numpy())
        label_hot = torch.from_numpy(label_hot)
        label_aux = torch.arange(args.closed_way, dtype=torch.int8).repeat(args.shot + args.query)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux, label_hot



    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        # start FSL training 
        label, label_aux, label_hot = self.prepare_label()

        for epoch in range(1, args.max_epoch+1):
            self.train_epoch+=1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()

            start_tm = time.time()
            for batch in self.train_loader:
                self.train_step+=1

                if torch.cuda.is_available():
                    data, gt_label = [_.cuda() for _ in batch]
                else:
                    data, gt_label = batch[0], batch[1]
                
                
                data_tm = time.time()
                self.dt.add(data_tm-start_tm)
                 # get saved centers
                if args.open_loss:
                    logits, open_logits, reg_logits = self.para_model(data)
                else:

                    logits, reg_logits = self.para_model(data)
                
                if reg_logits is not None:
                    if args.loss_type=='ce_loss':
                        loss = F.cross_entropy(logits, label)
                    elif args.loss_type=='edl_loss':
                        loss = select_edl_loss(logits, label_hot, epoch, self.args.closed_way, self.args)
                
                if args.open_loss:
                    Q = torch.ones_like(open_logits)
                    P = open_logits+1
                    open_loss = (P*(P/Q).log()).sum(axis = 1).mean()
                    total_loss = loss +args.balance * F.cross_entropy(reg_logits, label_aux)+args.open_loss_coeff*open_loss
                else:
                    total_loss = loss +args.balance * F.cross_entropy(reg_logits, label_aux)+
 
                tl2.add(total_loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)
                #print("Accuracy is", acc, "loss is", total_loss)
                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)    

                # refresh start_tm
                start_tm = time.time()

            self.lr_scheduler.step()
            self.try_evaluate(epoch)
            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')
        self.logger.dump()

   
    def get_snatcher_prob(self, logits, bproto, emb_dim, proto, query):
        snatch = []
        for j in range(logits.shape[0]):
            pproto = bproto.clone().detach()
            c = logits.argmax(1)[j]
            """Algorithm 1 Line 2"""
            pproto[0][c] = query.reshape(-1, emb_dim)[j]
            """Algorithm 1 Line 3"""
            pproto, _ = self.model.slf_attn(pproto, pproto, pproto)
            pdiff = (pproto-proto).pow(2).sum(-1).sum()/64.0
            """pdiff: d_SnaTCHer in Algorithm 1"""
            snatch.append(pdiff)
        return snatch


    def evaluate(self, data_loader):
        args = self.args
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 7))
        label = torch.arange(args.closed_way, dtype = torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        
        
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))

       
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                
                [instance_embs, support_idx, query_idx]  = self.model(data)
                support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
                query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))
                emb_dim = support.shape[-1]

                support = support[:, :, :args.closed_way].contiguous()
                # get mean of the support 
                bproto = support.mean(dim = 1)
                proto = bproto

                kquery = query[:, :, :args.closed_way].contiguous()
                uquery = query[:, :, args.closed_way:].contiguous()
                proto, _ = self.model.slf_attn(proto, proto, proto)

                klogits = -(kquery.reshape(-1, 1, emb_dim)-proto).pow(2).sum(2)/64.0
                ulogits = -(uquery.reshape(-1, 1, emb_dim)-proto).pow(2).sum(2)/64.0
                loss = F.cross_entropy(klogits, label)
                acc = count_acc(klogits, label)

                known_prob = F.softmax(klogits, 1).max(1)[0]
                unknown_prob = F.softmax(ulogits, 1).max(1)[0]
                known_scores = (known_prob).cpu().detach().numpy()
                unknown_scores = (unknown_prob).cpu().detach().numpy()
                known_scores = 1-known_scores
                unknown_scores = 1-unknown_scores
                auroc = calc_auroc(known_scores, unknown_scores)

                """Distance """
                kdist = -(klogits.max(1)[0])
                udist = -(ulogits.max(1)[0])
                kdist = kdist.cpu().detach().numpy()
                udist = udist.cpu().detach().numpy()
                dist_auroc = calc_auroc(kdist, udist)

                """Snatcher"""
                snatch_known = self.get_snatcher_prob(klogits, bproto, emb_dim, proto, kquery)
                snatch_unknown = self.get_snatcher_prob(ulogits, bproto, emb_dim, proto, uquery)
                pkdiff = torch.stack(snatch_known)
                pudiff = torch.stack(snatch_unknown)
                pkdiff = pkdiff.cpu().detach().numpy()
                pudiff = pudiff.cpu().detach().numpy()
                snatch_auroc = calc_auroc(pkdiff, pudiff)

                k_evidence = relu_evidence(-1/klogits)
                u_evidence = relu_evidence(-1/ulogits)
                k_alpha = k_evidence+1
                u_alpha = u_evidence+1
                k_s = torch.sum(k_alpha, axis = 1)
                u_s = torch.sum(u_alpha, axis = 1)
                k_uncert = args.closed_way/k_s
                u_uncert = args.closed_way/u_s
                edl_auroc = calc_auroc(k_uncert.cpu().detach().numpy(), u_uncert.cpu().detach().numpy())

                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                record[i-1, 2] = auroc
                record[i-1, 3] = dist_auroc
                record[i-1, 4] = snatch_auroc
                record[i-1, 5] = edl_auroc
                
               
        assert(i==record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        vaccm, vaccs = compute_confidence_interval(record[:, 1])
        vaucmp, vaucsp = compute_confidence_interval(record[:, 2])
        vaucmd, vaucsd = compute_confidence_interval(record[:, 3])
        vaucms, vaucss = compute_confidence_interval(record[:, 4])
        vaucmedl, vaucsedl = compute_confidence_interval(record[:, 5])

        return vl, vaccm, vaccs, vaucmp, vaucsp, vaucmd, vaucsd, vaucms, vaucss, vaucmedl, vaucsedl
    
    def evaluate_test(self):
        args = self.args 

        if args.open_loss:
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, str(self.args.shot)+'_s_'+str(self.args.run)+'_r_'+str(self.args.balance)+'_bal_'+str(args.open_loss_coeff)+'_olf_'+self.args.loss_type+'_'+'max_acc'+'.pth'))['params'])
        else:
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, str(self.args.shot)+'_s_'+str(self.args.run)+'_r_'+str(self.args.balance)+'_bal_'+self.args.loss_type+'_max_acc'+'.pth'))['params'])
        
        self.model.eval()



        record = np.zeros((1000, 7))
        label = torch.arange(args.closed_way, dtype = torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        
        
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))

        

       
       
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader, 1):
               
                if torch.cuda.is_available():
                    data, classes, image_names = batch
                    classes = classes.cuda()
                    data = data.cuda()
                    image_names = np.array(image_names)

                    #data, classes, image_names = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                
                
   
                [instance_embs, support_idx, query_idx]  = self.model(data)
                support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
                query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))
                emb_dim = support.shape[-1]

    
                support = support[:, :, :args.closed_way].contiguous()
                
                # get mean of the support 
                bproto = support.mean(dim = 1)
                proto = bproto

                kquery = query[:, :, :args.closed_way].contiguous()
                uquery = query[:, :, args.closed_way:].contiguous()

                proto_klogits = -(kquery.reshape(-1, 1, emb_dim)-bproto).pow(2).sum(2)/64.0
                protot_ulogits = -(uquery.reshape(-1, 1, emb_dim)-bproto).pow(2).sum(2)/64.0
               
                proto, _ = self.model.slf_attn(proto, proto, proto)

                klogits = -(kquery.reshape(-1, 1, emb_dim)-proto).pow(2).sum(2)/64.0
                ulogits = -(uquery.reshape(-1, 1, emb_dim)-proto).pow(2).sum(2)/64.0
                loss = F.cross_entropy(klogits, label)
                acc = count_acc(klogits, label)

                known_prob = F.softmax(klogits, 1).max(1)[0]
                unknown_prob = F.softmax(ulogits, 1).max(1)[0]
                known_scores = (known_prob).cpu().detach().numpy()
                unknown_scores = (unknown_prob).cpu().detach().numpy()
                known_scores = 1-known_scores
                unknown_scores = 1-unknown_scores
                auroc = calc_auroc(known_scores, unknown_scores)

                """Distance """
                kdist = -(klogits.max(1)[0])
                udist = -(ulogits.max(1)[0])
                kdist = kdist.cpu().detach().numpy()
                udist = udist.cpu().detach().numpy()
                dist_auroc = calc_auroc(kdist, udist)

                """Snatcher"""
                snatch_known = self.get_snatcher_prob(klogits, bproto, emb_dim, proto, kquery)
                snatch_unknown = self.get_snatcher_prob(ulogits, bproto, emb_dim, proto, uquery)
                pkdiff = torch.stack(snatch_known)
                pudiff = torch.stack(snatch_unknown)
                pkdiff = pkdiff.cpu().detach().numpy()
                pudiff = pudiff.cpu().detach().numpy()
                snatch_auroc = calc_auroc(pkdiff, pudiff)

                """Vacuity"""

                k_evidence = relu_evidence(-1/klogits)
                u_evidence = relu_evidence(-1/ulogits)
                k_alpha = k_evidence+1
                u_alpha = u_evidence+1
                k_s = torch.sum(k_alpha, axis = 1)
                u_s = torch.sum(u_alpha, axis = 1)
                k_uncert = args.closed_way/k_s
                u_uncert = args.closed_way/u_s
                edl_auroc = calc_auroc(k_uncert.cpu().detach().numpy(), u_uncert.cpu().detach().numpy())
            
                """EVR"""
                max_ev_k= k_alpha.max(1)[0]
                std_k = torch.std(k_alpha, axis = 1)
                max_ev_u = u_alpha.max(1)[0]
                std_u = torch.std(u_alpha, axis = 1)
                k_ratio = max_ev_k/std_k 
                u_ratio = max_ev_u/std_u
                mean_ratio = torch.mean(torch.cat([k_ratio, u_ratio]))

                hb_snatch_known = self.get_snatcher_prob(klogits, bproto, emb_dim, proto, kquery, mean_ratio)
                hb_snatch_unknown = self.get_snatcher_prob(ulogits, bproto, emb_dim, proto, uquery, mean_ratio)
                hb_pkdiff = torch.stack(hb_snatch_known)
                hb_pudiff = torch.stack(hb_snatch_unknown)
                hb_pkdiff = hb_pkdiff.cpu().detach().numpy()
                hb_pudiff = hb_pudiff.cpu().detach().numpy()
                evr_auroc = calc_auroc(hb_pkdiff, hb_pudiff)

                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                record[i-1, 2] = auroc
                record[i-1, 3] = dist_auroc
                record[i-1, 4] = snatch_auroc
                record[i-1, 5] = edl_auroc
                record[i-1, 6] = evr_auroc
                
                
        assert(i==record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        vaccm, vaccs = compute_confidence_interval(record[:, 1])
        vaucmp, vaucsp = compute_confidence_interval(record[:, 2])
        vaucmd, vaucsd = compute_confidence_interval(record[:, 3])
        vaucms, vaucss = compute_confidence_interval(record[:, 4])
        vaucmedl, vaucsedl = compute_confidence_interval(record[:, 5])
        vaucmevr, vaucsevr = compute_confidence_interval(record[:, 6])
        self.trlog['test_loss'] = vl 
        self.trlog['test_acc'] = float(vaccm)
        self.trlog['test_acc_interval'] = float(vaccs)
        self.trlog['test_auc_prob']= float(vaucmp)
        self.trlog['test_auc_prob_interval']= float(vaucsp)
        self.trlog['test_auc_dist']=float(vaucmd)
        self.trlog['test_auc_dist_interval']=float(vaucsd)
        self.trlog['test_auc_snatcher']= float(vaucms)
        self.trlog['test_auc_snatcher_interval']= float(vaucss)
        self.trlog['test_auc_edl']=float(vaucmedl)
        self.trlog['test_auc_edl_interval']=float(vaucsedl)
        self.trlog['test_auc_evr']=float(vaucmevr)
        self.trlog['test_auc_evr_interval']=float(vaucsevr)



        
    


    def final_record(self):
        # save the best performance in a txt file
        if self.args.open_loss:
            save_path = osp.join(self.args.save_path, str(self.args.shot)+'_s_'+str(self.args.run)+'_r_'+str(self.args.balance)+'_bal_'+str(self.args.open_loss_coeff)+'_olf_'+self.args.loss_type+'_openset_'+'{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval']))
        else:
            save_path = open(osp.join(self.args.save_path, str(self.args.shot)+'_s_'+str(self.args.run)+'_r_'+str(self.args.balance)+'_bal_'+self.args.loss_type+'_'+'{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval']))

        with(save_path , 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))   


  
    