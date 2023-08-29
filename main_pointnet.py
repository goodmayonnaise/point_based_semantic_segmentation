
import os, time
from datetime import datetime

from models.pointnet import PointNetDenseCls
from dataloaders.semantic_kitti import SemanticKITTI
from utils.pytorchtools import EarlyStopping
from train import Training

import torch
from torch.nn import DataParallel, NLLLoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def set_gpu():
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    torch.cuda.manual_seed_all(777)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    num_gpu = list(range(torch.cuda.device_count()))
    num_workers = len(gpus.split(",")) * 2

    return device, num_gpu, num_workers

def main():
    
    device, num_gpu, num_workers = set_gpu()
    
    # setting params
    nclasses = 20
    batch_size = len(num_gpu)*16
    epochs = 500
    name = ""
    
    # setting model
    model = PointNetDenseCls(k=nclasses, feature_transform=True) 
    model = DataParallel(model.cuda(), device_ids=num_gpu)
    optim = Adam(model.to(device).parameters())
    scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=100, eta_min=0.001)
    criterion = NLLLoss(ignore_index=0)
    from losses import feature_transform_regularizer
    criterion_feat = feature_transform_regularizer
    
    # setting data
    train_dataset = SemanticKITTI(nclasses=nclasses, mode='train')
    val_dataset = SemanticKITTI(nclasses=nclasses, mode='val')
    train_loader = DataLoader(train_dataset, batch_size, True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size, False, num_workers=num_workers)
    
    # setting save 
    configs = "batch{}_epoch{}_{}_{}".format(batch_size, epochs, str(criterion).split('(')[0], str(optim).split( )[0])
    print("Configs:", configs)
    now = time.strftime('%m%d_%H%M') 
    model_path = os.path.join("weights", configs, name+str(now))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    earlystop = EarlyStopping(patience=5, verbose=True, path=os.path.join(model_path, 'earlystop.pt'))

    metrics = {'t_loss':[], 'v_loss':[], 't_miou':[], 'v_miou':[]}

    if not os.path.exists(os.path.join(model_path, 'train')):
        os.makedirs(os.path.join(model_path, 'train'))
    if not os.path.exists(os.path.join(model_path, 'val')):
        os.makedirs(os.path.join(model_path, 'val'))
 
    writer_train = SummaryWriter(log_dir=os.path.join(model_path, 'train'))
    writer_val = SummaryWriter(log_dir = os.path.join(model_path, 'val'))

    with open(f'{model_path}/result.csv', 'a') as epoch_log:
        epoch_log.write('\nepoch\ttrain loss\tval loss\ttrain mIoU\tval mIoU')

    t_s = datetime.now()
    print(f'\ntrain start time : {t_s}')

    t = Training(model, epochs, train_loader, val_loader, optim, criterion, criterion_feat, nclasses,
                 scheduler, model_path, earlystop, device, metrics, writer_train, writer_val)
    t.train()
    print(f'\n[train time information]\n\ttrain start time\t{t_s}\n\tend of train\t\t{datetime.now()}\n\ttotal train time\t{datetime.now()-t_s}')

if __name__ == "__main__":
    
    main()
