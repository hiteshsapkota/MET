import torch
import torch.nn as nn
import numpy as np

class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        else:
            raise ValueError('')

    def split_instances(self, data):
        args = self.args
        if self.training:
            if args.open_loss:

                return  (torch.Tensor(np.arange((args.closed_way+args.open_way)*args.shot)).long().view(1, args.shot, args.closed_way+args.open_way), 
                     torch.Tensor(np.arange((args.closed_way+args.open_way)*args.shot, (args.closed_way+args.open_way) * (args.shot + args.query))).long().view(1, args.query, args.closed_way+args.open_way))
            else:

                 return  (torch.Tensor(np.arange((args.closed_way)*args.shot)).long().view(1, args.shot, args.closed_way), 
                            torch.Tensor(np.arange((args.closed_way)*args.shot, (args.closed_way) * (args.shot + args.query))).long().view(1, args.query, args.closed_way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way))

    def forward(self, x, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)
            num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)
            if self.training:
                if self.args.open_loss:
                    close_logits, open_logits, logits_reg= self._forward(instance_embs, support_idx, query_idx)
                    return close_logits, open_logits, logits_reg  
                else:

                    logits, logits_reg= self._forward(instance_embs, support_idx, query_idx)
                    return logits, logits_reg  
            else:
                logits = self._forward(instance_embs, support_idx, query_idx)
                return logits

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')