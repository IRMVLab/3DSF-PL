import sys, os
import os.path as osp
import numpy as np
import math
import torch.utils.data as data
from PIL import Image
import lib.utils.calibration as calibration
from torchvision import transforms


class KITTI_pytorch(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    # data_root: /data0/dataset
    def __init__(self, train, num_points, data_root, data_dir_list):
        self.root = osp.join(data_root, 'data_odometry_color','sequences')
        # assert train is False
        self.train = train
        self.npoints = num_points
        data_dir_list.sort()
        self.data_list = data_dir_list
        self.tag=np.random.choice([0])
        self.data_len_sequence=[4540, 1100, 4660, 800, 270, 2760, 1100, 1100, 4070, 1590, 1200]


        data_sum=[0]
        for i in self.data_list:
            data_sum.append(data_sum[-1]+self.data_len_sequence[i]+1)


        self.data_sum=data_sum

        # train set
        if self.train == 0:
            self.datapath = self.root
            # self.file_list = os.listdir(self.datapath)
            # self.file_list.sort()

        # Validation set
        elif self.train == 1:
            self.datapath = self.root

        # test set
        elif self.train == 2:
            self.datapath = self.root

    def __len__(self):
        return self.data_sum[-1]

    def __getitem__(self, index):  ################00

        tag = self.tag

        if self.train == 1:  ####### evalaute
            tag = 0
            
        sequence_str_list=[]
        for item in self.data_list:
            sequence_str_list.append('{:02d}'.format(item))

        # data sequence starting point
        if index in self.data_sum:
            index_index=self.data_sum.index(index)
            #pose_path='pose/'+sequence_str_list[index_index]+'_diff.txt'
            #pose=np.loadtxt(pose_path,dtype=float)
            datapath_right = os.path.join(self.datapath, sequence_str_list[index_index],'image_3')
            datapath_left = os.path.join(self.datapath, sequence_str_list[index_index],'image_2')
            depth = os.path.join(self.datapath, sequence_str_list[index_index],'depth_map')
            calib = os.path.join(self.datapath, sequence_str_list[index_index],'calib')
            self.file_list_right = os.listdir(datapath_right)
            self.file_list_right.sort()
            self.file_list_left = os.listdir(datapath_left)
            self.file_list_left.sort()
            self.depth_list = os.listdir(depth)
            self.depth_list.sort()
            self.calib_list = os.listdir(calib)
            self.calib_list.sort()
            index_=0
            fn1 = self.file_list_right[index_]
            fn2 = self.file_list_left[index_]
            fn3 = self.file_list_right[index_]
            fn4 = self.file_list_left[index_]
            d1 = self.depth_list[index_]
            d2 = self.depth_list[index_]
            c1 = self.calib_list[index_]
            c2 = self.calib_list[index_]
        # data sequence 
        else:
            
            index_index,data_begin,data_end=self.get_index(index,self.data_sum)
            #pose_path = 'pose/' + sequence_str_list[index_index] + '_diff.txt'
            #pose = np.loadtxt(pose_path, dtype=float)
            datapath_right = os.path.join(self.datapath, sequence_str_list[index_index],'image_3')
            datapath_left = os.path.join(self.datapath, sequence_str_list[index_index],'image_2')
            depth = os.path.join(self.datapath, sequence_str_list[index_index],'depth_map')
            calib = os.path.join(self.datapath, sequence_str_list[index_index],'calib')
            self.file_list_right = os.listdir(datapath_right)
            self.file_list_right.sort()
            self.file_list_left = os.listdir(datapath_left)
            self.file_list_left.sort()
            self.depth_list = os.listdir(depth)
            self.depth_list.sort()
            self.calib_list = os.listdir(calib)
            self.calib_list.sort()
            
            index_=index-data_begin
            fn1 = self.file_list_right[index_-1]
            fn2 = self.file_list_left[index_-1]
            fn3 = self.file_list_right[index_]
            fn4 = self.file_list_left[index_]
            d1 = self.depth_list[index_-1]
            d2 = self.depth_list[index_]
            c1 = self.calib_list[index_-1]
            c2 = self.calib_list[index_]
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        fn1 = os.path.join(datapath_right, fn1)
        fn2 = os.path.join(datapath_left, fn2)
        fn3 = os.path.join(datapath_right, fn3)
        fn4 = os.path.join(datapath_left, fn4)
        d1 = os.path.join(depth, d1)
        d2 = os.path.join(depth, d2)
        c1 = os.path.join(calib, c1)
        c2 = os.path.join(calib, c2)
        
        # ######################################################
        img1 = Image.open(fn1).convert('RGB')
        img2 = Image.open(fn2).convert('RGB')
        img3 = Image.open(fn3).convert('RGB')
        img4 = Image.open(fn4).convert('RGB')
        if transform is not None:
            img1 = transform(img1)
            img2 = transform(img2)
            img3 = transform(img3)
            img4 = transform(img4)

        W = img2.size()[2]
        H = img2.size()[1]

        depth1 = np.load(d1).astype(np.float32)
        depth2 = np.load(d2).astype(np.float32)
        top_pad = 384 - H
        right_pad = 1248 - W
        depth1 = np.pad(depth1, ((top_pad, 0), (0, right_pad)), 'constant', constant_values=0)
        depth2 = np.pad(depth2, ((top_pad, 0), (0, right_pad)), 'constant', constant_values=0)
        calib1 = calibration.Calibration(c1)
        calib2 = calibration.Calibration(c2)
        
        '''T_diff = pose[index_:index_ + 1, :]  ##### read the transform matrix
        T_diff = T_diff.reshape(3, 4)
        filler = np.array([0.0, 0.0, 0.0, 1.0])
        filler = np.expand_dims(filler, axis=0)  ##1*4
        T_diff = np.concatenate([T_diff, filler], axis=0)  # 4*4
        
        T_gt = T_diff
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3:]

        z_gt, y_gt, x_gt = self.mat2euler(M=R_gt)
        q_gt = self.euler2quat(z=z_gt, y=y_gt, x=x_gt)

        t_gt=t_gt.astype(np.float32)
        q_gt=q_gt.astype(np.float32)'''

        img_left = []
        img_left.append(img2)
        img_left.append(img4)
        img_right = []
        img_right.append(img1)
        img_right.append(img3)

        gt_depth =[]
        gt_depth.append(depth1)
        gt_depth.append(depth2)
        
        Calib = []
        Calib.append(calib1)
        Calib.append(calib2)
        
        batch = {'img_left': img_left, 'img_right': img_right,'gt_depth':gt_depth ,'calib':Calib }
        print('##########################################')
        return batch


    def get_index(self,value,mylist):
        mylist.sort()
        for i,num in enumerate(mylist):
            if num>value:
                return i-1,mylist[i-1],num

    def mat2euler(self, M, cy_thresh=None, seq='zyx'):

        M = np.asarray(M)
        if cy_thresh is None:
            cy_thresh = np.finfo(M.dtype).eps * 4

        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
        cy = math.sqrt(r33 * r33 + r23 * r23)
        if seq == 'zyx':
            if cy > cy_thresh:  # cos(y) not close to zero, standard form
                z = math.atan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
                y = math.atan2(r13, cy)  # atan2(sin(y), cy)
                x = math.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
            else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
                # so r21 -> sin(z), r22 -> cos(z) and
                z = math.atan2(r21, r22)
                y = math.atan2(r13, cy)  # atan2(sin(y), cy)
                x = 0.0
        elif seq == 'xyz':
            if cy > cy_thresh:
                y = math.atan2(-r31, cy)
                x = math.atan2(r32, r33)
                z = math.atan2(r21, r11)
            else:
                z = 0.0
                if r31 < 0:
                    y = np.pi / 2
                    x = math.atan2(r12, r13)
                else:
                    y = -np.pi / 2
        else:
            raise Exception('Sequence not recognized')
        return z, y, x

    def euler2quat(self, z=0, y=0, x=0, isRadian=True):
        ''' Return quaternion corresponding to these Euler angles
        Uses the z, then y, then x convention above
        Parameters
        ----------
        z : scalar
            Rotation angle in radians around z-axis (performed first)
        y : scalar
            Rotation angle in radians around y-axis
        x : scalar
            Rotation angle in radians around x-axis (performed last)
        Returns
        -------
        quat : array shape (4,)
            Quaternion in w, x, y z (real, then vector) format
        Notes
        -----
        We can derive this formula in Sympy using:
        1. Formula giving quaternion corresponding to rotation of theta radians
            about arbitrary axis:
            http://mathworld.wolfram.com/EulerParameters.html
        2. Generated formulae from 1.) for quaternions corresponding to
            theta radians rotations about ``x, y, z`` axes
        3. Apply quaternion multiplication formula -
            http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
            formulae from 2.) to give formula for combined rotations.
        '''

        if not isRadian:
            z = ((np.pi) / 180.) * z
            y = ((np.pi) / 180.) * y
            x = ((np.pi) / 180.) * x
        z = z / 2.0
        y = y / 2.0
        x = x / 2.0
        cz = math.cos(z)
        sz = math.sin(z)
        cy = math.cos(y)
        sy = math.sin(y)
        cx = math.cos(x)
        sx = math.sin(x)
        return np.array([
            cx * cy * cz - sx * sy * sz,
            cx * sy * sz + cy * cz * sx,
            cx * cz * sy - sx * cy * sz,
            cx * cy * sz + sx * cz * sy])


