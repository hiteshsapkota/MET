
# This repository contains the code for the paper "Meta Evidential Transformer for Few-Shot Open-Set Recognition".
## Dataset Information:

All datasets are publicly available. Please put the dataset as follow:

**1. data:**
        This contains 4 folders imagenet, tieredimagenet, cifar100, caltech. Please download data and put in corresponding folder      
**2. class_info:**
        Our approach has an unique way to split the data. Similar to data, it contains four folders each for dataset. It contains class index of openset and closeset classes    
**3. saves:**
        Contains initialization folder with four folders within each for each dataset. Those folders hold pretrained weights.

## Code Running Information:
To train the model run main.py with following parameters:
  
    --open_loss -> boolen telling whether to use explicit open set loss during training.
    --open_loss_coeff -> weights given for openset loss.
    --loss_type-> edl_loss or ce_loss used to train the model.
    --dataset -> dataset type use to train the model.
    --shot -> number of support set samples per task [1, 5].
    --query -> Number of samples per class in query set [15].

For other commands directly look at main.py

After training, the trained model will be stored under checkpoints. Also, all training losses, and validation accuracies are sotred in the same folder.
