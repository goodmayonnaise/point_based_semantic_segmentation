
import yaml
import numpy as np 

import torch
from torch.utils.data import Dataset


class SemanticKITTI(Dataset):
    def __init__(self, nclasses, mode, front=None, shape=(256, 1024)):
        CFG = self.load_config()
        self.swap_dict = CFG['learning_map']
        self.mode = mode
        self.path = self.data_path_load()

        self.nclasses = nclasses
        self.input_shape = shape     
        
        self.pcd_paths = self.path[0]
        self.label_paths = self.path[1]

    def load_config(self):
        cfg_path = 'dataloaders/configs/semantic-kitti.yaml'
        try:
            print("Opening config file %s" % cfg_path)
            CFG = yaml.safe_load(open(cfg_path, 'r'))
        except Exception as e:
            print(e)
            print("Error opening yaml file.")
            quit()
        return CFG
    
    def replace_with_dict(self, ar):
        # Extract out keys and values
        k = np.array(list(self.swap_dict.keys()))
        v = np.array(list(self.swap_dict.values()))

        # Get argsort indices
        sidx = k.argsort()
        
        # Drop the magic bomb with searchsorted to get the corresponding
        # places for a in keys (using sorter since a is not necessarily sorted).
        # Then trace it back to original order with indexing into sidx
        # Finally index into values for desired output.
        return v[sidx[np.searchsorted(k,ar,sorter=sidx)]]     

    def pcd_jitter(self, pcd, sigma=0.01, clip=0.05):
        N, C = pcd.shape
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip).astype(pcd.dtype)
        jittered_data += pcd
        return jittered_data

    def __len__(self):
        return len(self.pcd_paths)
    
    def per_class(self, t):

        per_class = torch.zeros([t.shape[0], self.nclasses, t.shape[1], t.shape[2]]).cuda()

        for i in range(self.nclasses):
            per_class[:,i] = torch.where(t==i, 1, 0)
        
        return per_class
    
    def __getitem__(self, idx):

        scan = np.fromfile(self.pcd_paths[idx], dtype=np.float32)
        scan = scan.reshape((-1, 4))[:,0:3]
        choice = np.random.choice(scan.shape[0], 2500, replace=True)
        
        scan = scan[choice,:]
        if self.mode == 'train' : # augmentation
            scan = self.pcd_jitter(scan) 
        
        scan = scan.transpose(1, 0)
        point = torch.FloatTensor(scan)
        
        label = np.fromfile(self.label_paths[idx], dtype=np.int32)
        label = label[choice]
        label = label & 0xFFF
        label = self.replace_with_dict(label)
        
        label = torch.FloatTensor(label.reshape((-1)))
        
        if label.shape[0] != point.shape[1]:
            print("Point shape: ", point.shape)
            print("Label shape: ", label.shape)
        
        return {'xyz':point, 'label':label}
        
    def data_path_load(self):
        pcd_paths = []
        label_paths = []
        
        if self.mode == 'train':
            f = open('dataloaders/configs/train_pcd.txt','r')
            lines = f.readlines()
            for line in lines:
                line = line[:-1]
                pcd_paths.append(line) 
                label_paths.append(line.replace('.bin', '.label').replace('velodyne', 'labels'))
        elif self.mode == 'val':
            f = open('dataloaders/configs/val_pcd.txt', 'r')
            lines = f.readlines()
            for line in lines:
                line = line[:-1]
                pcd_paths.append(line) 
                label_paths.append(line.replace('.bin', '.label').replace('velodyne', 'labels'))
        elif self.mode == 'test':
            f = open('dataloaders/configs/test_pcd.txt', 'r')
            lines = f.readlines()
            for line in lines:
                line = line[:-1]
                pcd_paths.append(line) 
                label_paths.append(line.replace('.bin', '.label').replace('velodyne', 'labels'))
            
        return pcd_paths, label_paths           


if __name__ == "__main__":

    train_dataset = SemanticKITTI(nclasses=20, mode='train')
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, 1, True, num_workers=0)

    for iter, batch in enumerate(train_loader):

        print()
