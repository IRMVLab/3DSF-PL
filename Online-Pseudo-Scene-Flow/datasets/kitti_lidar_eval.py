import sys, os
import os.path as osp
import numpy as np
import torch
import torch.utils.data as data

__all__ = ['KITTIlidar']


class KITTIlidar(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 remove_ground = True):
        self.root = osp.join('/dataset/sceneflow_eval_dataset', 'data/lidar_kitti/')
        #assert train is False
        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.remove_ground = remove_ground

        self.samples = self.make_dataset()
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded, flow = self.pc_loader(self.samples[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        '''if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)'''

        pc1_norm = pc1_loaded
        pc2_norm = pc2_loaded
        return pc1_loaded, pc2_loaded, pc1_norm, pc2_norm, flow, self.samples[index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

    def make_dataset(self):
        do_mapping = False
        root = osp.realpath(osp.expanduser(self.root))

        all_paths = sorted(os.walk(root))
        useful_paths = []
        c = 0
        for i in all_paths[0][2]:
            useful_paths.append(all_paths[0][0] + '/' + all_paths[0][2][c])
            c = c + 1
        #useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
        '''try:
            assert (len(useful_paths) == 200)
        except AssertionError:
            print('assert (len(useful_paths) == 200) failed!', len(useful_paths))'''

        if do_mapping:
            mapping_path = osp.join(osp.dirname(__file__), 'KITTI_mapping.txt')
            print('mapping_path', mapping_path)

            with open(mapping_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            useful_paths = [path for path in useful_paths if lines[int(osp.split(path)[-1])] != '']

        res_paths = useful_paths
        
        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        data = np.load(path)  #.astype(np.float32)
        pc1 = data['pc1'] # shape: pc1.shape != pc2.shape
        pc2 = data['pc2']
        flow = data['flow'] #pc1.shape = flow.shape

        pc1 = pc1.astype('float32')
        pc2 = pc2.astype('float32')
        flow = flow.astype('float32')
        if self.remove_ground:
            is_ground1 = pc1[:,1] > -1.42
            is_ground2 = pc2[:,1] > -1.42
            pc1 = pc1[is_ground1]
            flow = flow[is_ground1]
            pc2 = pc2[is_ground2]

            is_ground1 = pc1[:,2] < 35.
            is_ground2 = pc2[:,2] < 35.
            pc1 = pc1[is_ground1]
            flow = flow[is_ground1]
            pc2 = pc2[is_ground2]
        len1 = len(pc1)
        len2 = len(pc2)  
        if len1 < len2:
            indx = len1
        else:
            indx = len2
        idx = torch.randperm(indx)
        idx = idx.numpy().tolist()
        pc1 = pc1[idx]
        flow = flow[idx]
        pc2 = pc2[idx]

        # print('pc1:',pc1.shape, 'pc2', pc2.shape)
        #np.save('points/pc1_loaded{}'.format(0),pc1)
        return pc1, pc2, flow
