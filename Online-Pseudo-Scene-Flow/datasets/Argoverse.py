import glob
import numpy as np
from torch.utils.data import Dataset

__all__ = ['ArgoverseSceneFlowDataset']


class ArgoverseSceneFlowDataset(Dataset):
    def __init__(self, options, partition="val", width=1):
        self.options = options
        self.partition = partition
        self.width = width
        val_path = '/dataset/sceneflow_eval_dataset/argoverse/argoverse'
        if self.partition == "train":
            self.datapath = sorted(glob.glob(f"{self.options.dataset_path}/training/*/*/*"))
        elif self.partition == "test":
            self.datapath = sorted(glob.glob(f"{self.options.dataset_path}/testing/*/*/*"))
        elif self.partition == "val":
            self.datapath = sorted(glob.glob(f"{val_path}/val/*/*"))
            
    def __len__(self):
        return len(self.datapath)
    
    def __getitem__(self, index):
        filename = self.datapath[index]

        log_id = filename.split("/")[-3]
        dataset_dir = filename.split(log_id)[0]

        with open(filename, 'rb') as fp:
            data = np.load(fp)
            pc1 = data['pc1']
            pc2 = data['pc2']
            flow = data['flow']
            pc1 = pc1[:,[1,2,0]]
            pc2 = pc2[:,[1,2,0]]
            flow = flow[:,[1,2,0]]
            nw1,nw2 = list(np.where(pc1[:,2] > 0))[0],list(np.where(pc1[:,2] > 35))[0]
            w1 = np.setdiff1d(nw1, nw2)
            pc1 = pc1[w1]
            flow = flow[w1]
            nw1,nw2 = list(np.where(pc1[:,0] > -35))[0],list(np.where(pc1[:,0] > 35))[0]
            w1 = np.setdiff1d(nw1, nw2)
            pc1 = pc1[w1]
            flow = flow[w1]
            nw11,nw22 = list(np.where(pc2[:,2] > 0))[0],list(np.where(pc2[:,2] > 35))[0]
            w2 = np.setdiff1d(nw11, nw22)
            pc2 = pc2[w2]
            nw11,nw22 = list(np.where(pc2[:,0] > -35))[0],list(np.where(pc2[:,0] > 35))[0]
            w2 = np.setdiff1d(nw11, nw22)
            pc2 = pc2[w2]
            mask1_flow = data['mask1_tracks_flow']
            mask2_flow = data['mask2_tracks_flow']

        n1 = len(pc1)
        n2 = len(pc2)

        full_mask1 = np.arange(n1)
        full_mask2 = np.arange(n2)
        mask1_noflow = np.setdiff1d(full_mask1, mask1_flow, assume_unique=True)
        mask2_noflow = np.setdiff1d(full_mask2, mask2_flow, assume_unique=True)

        num_points = 8192
        nonrigid_rate = 0.8
        rigid_rate = 0.2
        if n1 >= num_points:
            '''if int(num_points * nonrigid_rate) > len(mask1_flow):
                num_points1_flow = len(mask1_flow)
                num_points1_noflow = num_points - num_points1_flow
            else:
                num_points1_flow = int(num_points * nonrigid_rate)
                num_points1_noflow = int(num_points * rigid_rate) + 1

            try:  # ANCHOR: argoverse has some cases without nonrigid flows.
                sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=False)
            except:
                sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=True)
            sample_idx1_flow = np.random.choice(mask1_flow, num_points1_flow, replace=False)
            sample_idx1 = np.hstack((sample_idx1_flow, sample_idx1_noflow))'''
            sample_idx1 = np.random.choice(full_mask1, num_points, replace=False)

            pc1_ = pc1[sample_idx1, :]
            #np.save('points/pc1_loaded{}'.format(index),pc1_)
            flow_ = flow[sample_idx1, :]

            pc1 = pc1_.astype('float32')
            flow = flow_.astype('float32')

        if n2 >= num_points:
            '''if int(num_points * nonrigid_rate) > len(mask2_flow):
                num_points2_flow = len(mask2_flow)
                num_points2_noflow = num_points - num_points2_flow
            else:
                num_points2_flow = int(num_points * nonrigid_rate)
                num_points2_noflow = int(num_points * rigid_rate) + 1
                
            try:  # ANCHOR: argoverse has some cases without nonrigid flows.
                sample_idx2_noflow = np.random.choice(mask2_noflow, num_points2_noflow, replace=False)
            except:
                sample_idx2_noflow = np.random.choice(mask2_noflow, num_points2_noflow, replace=True)
            sample_idx2_flow = np.random.choice(mask2_flow, num_points2_flow, replace=False)
            sample_idx2 = np.hstack((sample_idx2_flow, sample_idx2_noflow))'''
            sample_idx2 = np.random.choice(full_mask2, num_points, replace=False)

            pc2_ = pc2[sample_idx2, :]
            pc2 = pc2_.astype('float32')

        return pc1, pc2, pc1, pc2, flow, filename