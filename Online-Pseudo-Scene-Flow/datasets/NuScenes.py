import glob
import numpy as np
from torch.utils.data import Dataset

__all__ = ['NuScenesSceneFlowDataset']


class NuScenesSceneFlowDataset(Dataset):
    def __init__(self, options, partition="val", width=1):
        self.options = options
        self.partition = partition
        self.width = width
        val_path = '/dataset/sceneflow_eval_dataset/nuscenes/nuscenes'
        if self.partition == "train":
            self.datapath = sorted(glob.glob(f"{self.options.dataset_path}/train/*"))
        elif self.partition == "val":
            self.datapath = sorted(glob.glob(f"{val_path}/val/*"))
            
        # Bad data. Pretty noisy samples.
        bad_data = ["d02c6908713147e9a4ac5d50784815d3",
                    "ae989fac82d248b98ce769e753f60f87",
                    "365e72358ddb405e953cdad865815966",
                    "4a16cf07faf54dbf93e0c4c083b38c63",
                    "44fd8959bd574d7fb6773a9fe341282e",
                    "c6879ea1c3d845eebd7825e6e454bee1",
                    "359023a812c24fbcae41334842672dd2",
                    "aa5d89b9f988450eaa442070576913b7",
                    "c4344682d52f4578b5aa983612764e9b",
                    "6fd5607c93fa4b569eb2bd0d7f30f9a0",
                    "5ab1e1f0829541269856edca0f7517da",
                    "1c01cb36784e44fc8a5ef7d9689ef2fd",
                    "15a106fd45604b6bb85d67c1e5033022",
                    "6803c2feca4b40e78434cf209ee8c2da",
                    "6737346ecd5144d28cef656f17953959",
        ]
        self.datapath = [d for d in self.datapath if not any(bad in d for bad in bad_data)]

    def __getitem__(self, index):
        filename = self.datapath[index]

        with open(filename, 'rb') as fp:
            data = np.load(fp)
            pc1 = data['pc1'].astype('float32')
            pc2 = data['pc2'].astype('float32')
            flow = data['flow'].astype('float32')
            pc1 = pc1[:,[1,2,0]]
            pc2 = pc2[:,[1,2,0]]
            flow = flow[:,[1,2,0]]
            nw1,nw2 = list(np.where(pc1[:,2] > 0))[0],list(np.where(pc1[:,2] > 35))[0]
            w1 = np.setdiff1d(nw1, nw2)
            now1 = np.setdiff1d(np.arange(len(pc1)), w1)
            # w1 = list(set(nw1.tolist()+nw2.tolist()))
            pc1 = pc1[w1]
            flow = flow[w1]
            nw1,nw2 = list(np.where(pc1[:,0] > -35))[0],list(np.where(pc1[:,0] > 35))[0]
            w1 = np.setdiff1d(nw1, nw2)
            pc1 = pc1[w1]
            flow = flow[w1]
            nw11,nw22 = list(np.where(pc2[:,2] > 0))[0],list(np.where(pc2[:,2] > 35))[0]
            # w2 = list(set(nw11.tolist()+nw22.tolist()))
            w2 = np.setdiff1d(nw11, nw22)
            now2 = np.setdiff1d(np.arange(len(pc2)), w2)
            pc2 = pc2[w2]
            nw11,nw22 = list(np.where(pc2[:,0] > -35))[0],list(np.where(pc2[:,0] > 35))[0]
            w2 = np.setdiff1d(nw11, nw22)
            pc2 = pc2[w2]

            mask1_flow = data['mask1_tracks_flow']
            # mask1_flow = list(set(list(mask1_flow) +w1))
            mask1_flow = np.setdiff1d(mask1_flow, now1)
            mask2_flow = data['mask2_tracks_flow']
            # mask2_flow = list(set(list(mask2_flow) +w2))
            mask2_flow = np.setdiff1d(mask2_flow, now2)

        n1 = len(pc1)
        n2 = len(pc2)
        print('&&&&&&n1,n2&&&&&',n1,n2,len(mask1_flow),len(mask2_flow))
        full_mask1 = np.arange(n1)
        full_mask2 = np.arange(n2)
        mask1_noflow = np.setdiff1d(full_mask1, mask1_flow, assume_unique=True)
        mask2_noflow = np.setdiff1d(full_mask2, mask2_flow, assume_unique=True)
        use_all_points = False
        if use_all_points:
            num_points = 4096
        else:
            num_points = 4096
            nonrigid_rate = 0.8
            rigid_rate = 0.2
            if n1 >= num_points:
                '''if int(num_points * nonrigid_rate) > len(mask1_flow):
                    num_points1_flow = len(mask1_flow)
                    num_points1_noflow = num_points - num_points1_flow
                else:
                    num_points1_flow = int(num_points * nonrigid_rate)
                    num_points1_noflow = int(num_points * rigid_rate) + 1
                sample_idx1_flow = np.random.choice(mask1_flow, num_points1_flow, replace=False)
                try:  # ANCHOR: nuscenes has some cases without nonrigid flows.
                    sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=False)
                except:
                    sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=True)
                sample_idx1 = np.hstack((sample_idx1_flow, sample_idx1_noflow))'''
                sample_idx1 = np.random.choice(full_mask1, num_points, replace=False)
            else:
                sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, num_points - n1, replace=True)), axis=-1)
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
                sample_idx2_flow = np.random.choice(mask2_flow, num_points2_flow, replace=False)
                sample_idx2_noflow = np.random.choice(mask2_noflow, num_points2_noflow, replace=False)
                sample_idx2 = np.hstack((sample_idx2_flow, sample_idx2_noflow))'''
                sample_idx2 = np.random.choice(full_mask2, num_points, replace=False)
            else:
                sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, num_points - n2, replace=True)), axis=-1)

            pc2_ = pc2[sample_idx2, :]
            pc2 = pc2_.astype('float32')

        return pc1, pc2, pc1, pc2, flow, filename

    def __len__(self):
        return len(self.datapath)