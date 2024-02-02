import torch
import numpy as np
np.random.seed(1)

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, cluster_info, split_type):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.cluster_info = cluster_info
        self.split_type = split_type

        

        label = np.array(label)
        unique_labels = list(set(list(label)))
        
        self.m_ind = {}
        for i in unique_labels:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind

        
        self.open_classes = np.array([k for k, v in self.cluster_info.items() if v==1])
        self.closed_classes = np.array([k for k, v in self.cluster_info.items() if v==0])
    
        self.all_classes = np.concatenate([self.open_classes, self.closed_classes])

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []

            if self.split_type=='train' and not self.args.open_loss:

                shuffle_all_classes = np.random.permutation(self.all_classes)
                classes = shuffle_all_classes[:self.n_cls]
            else:
                shuffle_open_class = np.random.permutation(self.open_classes)
                shuffle_close_class = np.random.permutation(self.closed_classes)
                sel_close_cls = shuffle_close_class[:int(self.n_cls/2)]
                sel_open_cls = shuffle_open_class[:int(self.n_cls/2)]
                classes = np.concatenate([sel_close_cls, sel_open_cls])
            
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class RandomSampler():

    def __init__(self, label, n_batch, n_per):
        self.n_batch = n_batch
        self.n_per = n_per
        self.label = np.array(label)
        self.num_label = self.label.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = torch.randperm(self.num_label)[:self.n_per]
            yield batch
            
            
# sample for each class
class ClassSampler():

    def __init__(self, label, n_per=None):
        self.n_per = n_per
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return len(self.m_ind)

    def __iter__(self):
        classes = torch.arange(len(self.m_ind))
        for c in classes:
            l = self.m_ind[int(c)]
            if self.n_per is None:
                pos = torch.randperm(len(l))
            else:
                pos = torch.randperm(len(l))[:self.n_per]
            yield l[pos]
            
            
# for ResNet Fine-Tune, which output the same index of task examples several times
class InSetSampler():

    def __init__(self, n_batch, n_sbatch, pool): # pool is a tensor
        self.n_batch = n_batch
        self.n_sbatch = n_sbatch
        self.pool = pool
        self.pool_size = pool.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = self.pool[torch.randperm(self.pool_size)[:self.n_sbatch]]
            yield batch