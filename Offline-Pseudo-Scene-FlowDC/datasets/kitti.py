import sys, os
import os.path as osp
import numpy as np
import torch
import torch.utils.data as data
import open3d

__all__ = ['KITTI']

class KITTI(data.Dataset):
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
                 Stereo,
                 remove_ground = True):
        self.root = data_root
        #assert train is False
        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.remove_ground = remove_ground
        #print('self.root',self.root)
        self.samples, self.samples_depth = self.make_dataset()
        self.PCnormals = False
        self.Stereo = Stereo
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)-1

    def __getitem__(self, index):
        
        index2 = index+1
        #print(self.samples[index],self.samples[index2],index,'---',index2)
        pc1_loaded, pc2_loaded, pc1_normals, pc2_normals, depth2_loaded = self.pc_loader(self.samples[index],self.samples[index2], self.samples_depth[index2])
        # np.save('points/nosparse_{}'.format(index2), pc2_loaded)
        # np.save('points/nosparse_{}'.format(index), pc1_loaded)
        #np.save('points/rgbnosparse_{}'.format(index),pc1_loaded)
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        #np.save('points/rgbsparse_{}'.format(index2),pc2_transformed)
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)
        pc1 = pc1_transformed
        pc2 = pc2_transformed
        # if self.PCnormals:
        #     pc1_norm = pc1_normals.astype(np.float32)
        #     pc2_norm = pc2_normals.astype(np.float32)
        # else:
        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
            
        #np.save('points/rgbsparse_{}'.format(index),pc1)
        #np.save('points/rgbsparse_{}'.format(index2),pc2)
        #print('===----===',pc1.shape,pc2_norm.shape)
        return pc1, pc2, pc1_norm, pc2_norm, sf_transformed, depth2_loaded, self.samples[index]

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

        root = osp.realpath(osp.expanduser(self.root))
        
        all_paths = sorted(os.walk(root))[0][2]
        all_paths = sorted(all_paths)
        rootd = osp.realpath(osp.expanduser('../../dataset/data_odometry_color/sequences/02/Stereo_depth'))
        
        all_pathsd = sorted(os.walk(rootd))[0][2]
        all_pathsd = sorted(all_pathsd)
        return all_paths, all_pathsd
        
    def pc_loader(self, path11,path22, path_depth2):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        path1= osp.join(self.root,path11)
        path2= osp.join(self.root,path22)
        if self.Stereo:
            path_depth2= osp.join('../../dataset/data_odometry_color/sequences/02/depth_map', path22)
        else:
            path_depth2= osp.join('../../dataset/data_odometry_color/sequences/02/depth_map', path22)
        pc1 = np.load(osp.join(path1))  #.astype(np.float32)
        pc2 = np.load(osp.join(path2))
        # depth2 = pc2
        depth2 = np.load((osp.join(path_depth2)))

# =============================================================================
#         pc1 = np.fromfile(path1, dtype=np.float32).reshape(-1, 4)
#         pc2 = np.fromfile(path2, dtype=np.float32).reshape(-1, 4)
# =============================================================================
        if pc1.shape[-1] == 4:
            pc1 = np.delete(pc1,-1,axis=1)
            pc2 = np.delete(pc2,-1,axis=1)
        #pc1 = pc1[:,[1,2,0]]
        #pc2 = pc2[:,[1,2,0]]
        #pc1 = np.load(osp.join(path))  #.astype(np.float32)
        #pc2 = np.load(osp.join(path))  #.astype(np.float32)
        def fliter_radius(pc_data):
            pcl = open3d.geometry.PointCloud()
            pcl.points = open3d.utility.Vector3dVector(pc_data)
            cl, ind = pcl.remove_statistical_outlier(nb_neighbors=8, std_ratio=2)
            cl_arr=np.asarray(cl.points)
            return cl_arr,ind
        def estimate_pc_normals(pc_data):
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(pc_data)
            radius = 0.1
            max_nn = 30
            pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
            normal_point = open3d.utility.Vector3dVector(pcd.normals)
            pc_normals = np.asarray(normal_point)
            return pc_normals * 25
        if self.Stereo:
            is_ground1 = pc1[:,1] < 0.5
            pc1 = pc1[is_ground1]
            is_ground2 = pc2[:,1] < 0.5
            pc2 = pc2[is_ground2]
            
            is_ground1 = pc1[:,1] > -1.42
            pc1 = pc1[is_ground1]
            is_ground2 = pc2[:,1] > -1.42
            pc2 = pc2[is_ground2]
            
            is_wh10 = pc1[:,0] > -11
            is_wh11 = pc2[:,0] > -11
            pc1 = pc1[is_wh10]
            pc2 = pc2[is_wh11]
            is_wh11 = pc1[:,0] < 11
            is_wh21 = pc2[:,0] < 11
            pc1 = pc1[is_wh11]
            pc2 = pc2[is_wh21]
            len1 = len(pc1)
            len2 = len(pc2)  
            if len1 < len2:
                indx = len1
            else:
                indx = len2
            idx = torch.randperm(indx)
            idx = idx.numpy().tolist()
            pc1 = pc1[idx]
            pc2 = pc2[idx]
            pc1, pc2, depth2 = pc1.astype(np.float32), pc2.astype(np.float32), depth2.astype(np.float32)           
            return pc1, pc2, pc1, pc2, depth2
            
        else:
            pc1 = pc1[:,[1,2,0]]
            pc2 = pc2[:,[1,2,0]]
            len1 = len(pc1)
            len2 = len(pc2)  
            if len1 < len2:
                indx = len1
            else:
                indx = len2
            idx = torch.randperm(indx)
            idx = idx.numpy().tolist()
            pc1 = pc1[idx]
            pc2 = pc2[idx]

            # is_ground = np.logical_or(pc1[:,1] < -1.4, pc2[:,1] < -1.4)
            # not_ground = np.logical_not(is_ground)
            # pc1 = pc1[not_ground]
            # pc2 = pc2[not_ground]
            is_ground1 = pc1[:,1] < 0.5
            pc1 = pc1[is_ground1]
            is_ground2 = pc2[:,1] < 0.5
            pc2 = pc2[is_ground2]
            
            is_ground1 = pc1[:,1] > -1.40
            pc1 = pc1[is_ground1]
            is_ground2 = pc2[:,1] > -1.40
            pc2 = pc2[is_ground2]
            
            is_depth10 = pc1[:,2] > 0
            pc1 = pc1[is_depth10]
            is_depth20 = pc2[:,2] > 0
            pc2 = pc2[is_depth20]
            
            is_wh10 = pc1[:,0] > -10
            is_wh11 = pc2[:,0] > -10
            pc1 = pc1[is_wh10]
            pc2 = pc2[is_wh11]
            is_wh11 = pc1[:,0] < 10
            is_wh21 = pc2[:,0] < 10
            pc1 = pc1[is_wh11]
            pc2 = pc2[is_wh21]
            '''pc1 = pc1[:,[1,2,0]]
            pc2 = pc2[:,[1,2,0]]
            #pc1 = pc1[:,[1,2,0,3,4,5]]
            #pc2 = pc2[:,[1,2,0,3,4,5]]
            print('lennnnnnn2',len(pc1),len(pc2))
            near_mask_fov1 = np.logical_and(pc1[:, 2] > pc1[:, 0], pc1[:, 2] > -pc1[:, 0])
            near_mask_fov2 = np.logical_and(pc2[:, 2] > pc2[:, 0], pc2[:, 2] > -pc2[:, 0])
            pc1 = pc1[near_mask_fov1]
            pc2 = pc2[near_mask_fov2]
            print('lennnnnnn1',len(pc1),len(pc2))'''
            _,indxx1 = fliter_radius(pc1)
            _,indxx2 = fliter_radius(pc2)
            pc1 = pc1[indxx1]
            pc2 = pc2[indxx2]
            len1 = len(pc1)
            len2 = len(pc2)  
            if len1 < len2:
                indx = len1
            else:
                indx = len2
            idx = torch.randperm(indx)
            idx = idx.numpy().tolist()
            pc1 = pc1[idx]
            pc2 = pc2[idx]
            # if self.PCnormals:
            #     pc1_normals = estimate_pc_normals(pc1)
            #     pc2_normals = estimate_pc_normals(pc2)
            
            # #save_pcl(pc2, 'points/nosparse_{}'.format(1))
            # if self.PCnormals:    
            #     return pc1, pc2, pc1_normals, pc2_normals
            # else:
            pc1, pc2, depth2 = pc1.astype(np.float32), pc2.astype(np.float32), depth2.astype(np.float32)
            return pc1, pc2, pc1, pc2, depth2
