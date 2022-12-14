import numpy as np
import os
import open3d


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype = np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype = np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype = np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype = np.float32)

    return { 'P2'         : P2.reshape(3, 4),
             'P3'         : P3.reshape(3, 4),
             'R0'         : R0.reshape(3, 3),
             'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4) }


class Calibration(object):
    def __init__(self, calib_file):
        if isinstance(calib_file, str):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4
        self.C2V = inverse_rigid_trans(self.V2C)

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        # self.fu, self.fv, self.cu, self.cv = 868.993378, 866.063001, 525.942323, 420.042529
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)
        self.FB = 0.5372 * self.fu
    def disp_to_depth(self, disp_pred):
        depth_map = self.FB / disp_pred
        return depth_map
    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype = np.float32)))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_lidar: (N, 3)
        """
        pts_hom = self.cart_to_hom(np.dot(pts_rect, np.linalg.inv(self.R0.T)))
        pts_rect = np.dot(pts_hom, self.C2V.T)
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu #+ self.tx
        y = ((v - self.cv) * depth_rect) / self.fv #+ self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis = 1)
        return pts_rect

    def depthmap_to_rect(self, depth_map):
        """
        :param depth_map: (H, W), depth_map
        :return:
        """
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs

    def depthmap_to_lidar(self, depth_map):
         
        depth_map = self.FB / depth_map
        
        pts_rect, _, _ = self.depthmap_to_rect(depth_map)
        lidarpc = self.rect_to_lidar(pts_rect)
        lidarpc = processing_generated(lidarpc)
        return lidarpc


    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis = 2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis = 1), np.min(y, axis = 1)
        x2, y2 = np.max(x, axis = 1), np.max(y, axis = 1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis = 1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis = 2)

        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        """
        Can only process valid u, v, d, which means u, v can not beyond the image shape, reprojection error 0.02
        :param u: (N)
        :param v: (N)
        :param d: (N), the distance between camera and 3d points, d^2 = x^2 + y^2 + z^2
        :return:
        """
        assert self.fu == self.fv, '%.8f != %.8f' % (self.fu, self.fv)
        fd = np.sqrt((u - self.cu) ** 2 + (v - self.cv) ** 2 + self.fu ** 2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d ** 2 - x ** 2 - y ** 2)
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis = 1)
        return pts_rect

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

def fliter_radius(pc_data):
            pcl = open3d.geometry.PointCloud()
            pcl.points = open3d.utility.Vector3dVector(pc_data)
            cl, ind = pcl.remove_statistical_outlier(nb_neighbors=8, std_ratio=2)
            cl_arr=np.asarray(cl.points)
            return cl_arr,ind


# def processing_generated(PC):
#     pc1 = PC[:,[1,2,0]]
#     is_ground1 = pc1[:,1] < 1.5
#     pc1 = pc1[is_ground1]
#     is_ground1 = pc1[:,1] > -1.42
#     pc1 = pc1[is_ground1]
    
#     is_depth10 = pc1[:,2] > 0
#     pc1 = pc1[is_depth10]
#     is_depth10 = pc1[:,2] < 40
#     pc1 = pc1[is_depth10]
    
#     is_wh10 = pc1[:,0] > -13
#     pc1 = pc1[is_wh10]
#     is_wh11 = pc1[:,0] < 13
#     pc1 = pc1[is_wh11]
#     _,indxx1 = fliter_radius(pc1)
#     pc1 = pc1[indxx1]
#     return pc1


def processing_generated(PC):
    pc1 = PC[:,[1,2,0]]
    is_ground1 = pc1[:,1] < 2.0
    pc1 = pc1[is_ground1]
    is_ground1 = pc1[:,1] > -1.6
    pc1 = pc1[is_ground1]
    
    is_depth10 = pc1[:,2] > 0
    pc1 = pc1[is_depth10]
    is_depth10 = pc1[:,2] < 72
    pc1 = pc1[is_depth10]
    
    is_wh10 = pc1[:,0] > -40
    pc1 = pc1[is_wh10]
    is_wh11 = pc1[:,0] < 40
    pc1 = pc1[is_wh11]
    _,indxx1 = fliter_radius(pc1)
    pc1 = pc1[indxx1]
    pc1 = pc1[:,[2,0,1]]
    return pc1