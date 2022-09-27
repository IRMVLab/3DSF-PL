import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch_scatter import scatter_max
import os.path as osp
import math
import random
import numpy as np
from depth_network import logger
import os
import shutil
from depth_network.models import *
import lib.net.fliter as FR

#import kitti_util
import lib.net.kitti_util as kitti_util
import lib.net.batch_utils as batch_utils
#import batch_utils

from PIL import Image
from tensorboardX import SummaryWriter


def loader(path):
    return Image.open(path).convert('RGB')


def dynamic_baseline(calib):
    P3 = calib.P3
    P = calib.P2
    baseline = P3[0, 3] / (-P3[0, 0]) - P[0, 3] / (-P[0, 0])
    return baseline


class DepthModel():
    def __init__(self, maxdisp, down, maxdepth, pretrain, save_tag, mode='TRAIN', dynamic_bs=False,
                     lr=0.001, mgpus=False, lr_stepsize=[10, 20], lr_gamma=0.1):

        result_dir = os.path.join('../', 'output', 'depth', save_tag)
        # set logger
        log = logger.setup_logger(os.path.join(result_dir, 'training.log'))

        # set tensorboard
        writer = SummaryWriter(result_dir + '/tensorboardx')

        model = stackhourglass(maxdisp, down=down, maxdepth=maxdepth)

        # Number of parameters
        log.info('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))
        if mgpus or mode == 'TEST':
            model = nn.DataParallel(model)
        model = model.cuda()

        torch.backends.cudnn.benchmark = False

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        scheduler = MultiStepLR(
            optimizer, milestones=lr_stepsize, gamma=lr_gamma)

        if pretrain is not None:
            if os.path.isfile(pretrain):
                log.info("=> loading pretrain '{}'".format(pretrain))
                checkpoint = torch.load(pretrain)
                if mgpus or mode == 'TEST':
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(self.strip_prefix(checkpoint['state_dict']))
                optimizer.load_state_dict(checkpoint['optimizer'])

            else:
                log.info(
                    '[Attention]: Do not find checkpoint {}'.format(pretrain))

        optimizer.param_groups[0]['lr'] = lr

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = model
        self.dynamic_bs = dynamic_bs
        self.mode = mode
        self.result_dir = result_dir

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def load_data(self, batch_left_img, batch_right_img, batch_gt_depth, batch_calib):
        left_imgs, right_imgs, calibs = [], [], []
        for left_img, right_img, calib in zip(
                batch_left_img, batch_right_img, batch_calib):
            if self.dynamic_bs:
                calib = calib.P2[0, 0] * dynamic_baseline(calib)
            else:
                calib = calib.P2[0, 0] * 0.54

            calib = torch.tensor(calib)
            left_img = self.img_transform(left_img)
            right_img = self.img_transform(right_img)

            # pad to (384, 1248)
            C, H, W = left_img.shape
            top_pad = 384 - H
            right_pad = 1248 - W
            left_img = F.pad(
                left_img, (0, right_pad, top_pad, 0), "constant", 0)
            right_img = F.pad(
                right_img, (0, right_pad, top_pad, 0), "constant", 0)

            left_imgs.append(left_img)
            right_imgs.append(right_img)
            calibs.append(calib)

        left_img = torch.stack(left_imgs)
        right_img = torch.stack(right_imgs)
        calib = torch.stack(calibs)

        gt_depth = torch.from_numpy(batch_gt_depth).cuda(non_blocking=True)

        return left_img.float(), right_img.float(), gt_depth.float(), calib.float()


    def train(self, batch, start=2.0, max_high=1.0):
        imgL, imgR, gt_depth, calib = self.load_data(
            batch['left_image'], batch['right_image'], batch['gt_depth'], batch['calib'])
        imgL, imgR, gt_depth, calib = imgL.cuda(), imgR.cuda(), gt_depth.cuda(), calib.cuda()

        # ---------
        mask = (gt_depth >= 1) * (gt_depth <= 80)
        mask.detach_()
        #print('mask', torch.sum(mask).float()/(mask.size()[0]*mask.size()[1]*mask.size()[2]))
        # ----

        output1, output2, output3 = self.net(imgL, imgR, calib)
        output3 = torch.squeeze(output3, 1)
        
        
        #save_pcl(output3[0], 'depth/depth_{}'.format(batch['sample_id'][0]))
        
        #save_pcl(output3[1], 'depth/depth_{}'.format(batch['sample_id'][1]))

        def hook_fn(grad):
            print(grad.size())
            a = (grad == 0).float()
            rate = 100 * torch.sum(a) / (grad.size()[0] * grad.size()[1] * grad.size()[2])
            print('depth_map', rate, torch.mean(grad)/(rate/100), torch.max(grad), torch.min(grad))
            print('one_norm', torch.sum(torch.abs(grad)))

        loss = 0.5 * F.smooth_l1_loss(output1[mask], gt_depth[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
            output2[mask], gt_depth[mask], size_average=True) + F.smooth_l1_loss(output3[mask], gt_depth[mask],
                                                                                size_average=True)

        points = []
        for depth, calib_info, image, sample_id in zip(
                output3, batch['calib'], batch['left_image'], batch['sample_id']):
            calib_info = kitti_util.Calib(calib_info)
            W, H = image.size
            print(' H,W',H,W)
            depth = depth[-H:, :W]
            print(depth)
            
            cloud = depth_to_pcl(calib_info, depth, max_high=max_high)
            
            cloud = filter_cloud(cloud, image, calib_info)
            
            #cloud = transform(cloud, calib_info, sparse_type='angular_min', start=2.0)
            #save_pcl(cloud, 'point1/nosparse_{}'.format(sample_id))
            points.append(cloud)

        point1 = points[:2]
        #point2 = points[2:]
        
        out1, out2 = removep(point1,sample_id-2)
        #out3, out4 = removep(point2,sample_id)
        #points1 = torch.stack((out1,out3), dim=0)
        #points2 = torch.stack((out2,out4), dim=0)
        return loss, out1.unsqueeze(0), out2.unsqueeze(0)
        #return loss, points1, points2
    def eval(self, batch, max_high=1.0):
        imgL, imgR, gt_depth, calib = self.load_data(
            batch['left_image'], batch['right_image'], batch['gt_depth'], batch['calib'])
        imgL, imgR, gt_depth, calib = imgL.cuda(), imgR.cuda(), gt_depth.cuda(), calib.cuda()
        # ---------
        mask = (gt_depth >= 1) * (gt_depth <= 80)
        mask.detach_()
        #print('mask', torch.sum(mask).float() / (mask.size()[0] * mask.size()[1] * mask.size()[2]))
        # ----
        
        with torch.no_grad():
            output3 = self.net(imgL, imgR, calib)
            output3 = torch.squeeze(output3, 1)
            #loss = F.smooth_l1_loss(output3[mask], gt_depth[mask], size_average=True)
            #save_pcl(output3, 'points/depth_{}'.format(batch['sample_id']))
            loss = 0
            '''calib_root = 'utils/calib_cam_to_cam/'
            calib_path = osp.join(calib_root, '000003' + '.txt')
            with open(calib_path) as fd:
                lines = fd.readlines()
                assert len([line for line in lines if line.startswith('P_rect_02')]) == 1
                P_rect_left = \
                    np.array([float(item) for item in
                              [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                             dtype=np.float32).reshape(3, 4)

            assert P_rect_left[0, 0] == P_rect_left[1, 1]'''
            
            points = []
            for depth, calib_info, image, gt_lidar, sample_id in zip(
                    output3, batch['calib'], batch['left_image'], batch['gt_depth'], batch['sample_id']):
                calib_info = kitti_util.Calib(calib_info)
                W, H = image.size
                #gt_lidar = torch.tensor(gt_lidar, dtype=torch.float32).cuda()
                #datagdc = FR.FliterGDC(depth)
                #depth = datagdc.DepthCorrection(depth,gt_lidar,calib_info) *******************************
                #depth = torch.tensor(depth, dtype=torch.float32).cuda()
                
                depth = depth[-H:, :W]
                
                #cloud = pixel2xyz(depth, P_rect_left)
                #save_pcl(cloud, 'point1/nosparse_{}'.format(sample_id))
                cloud = depth_to_pcl(calib_info, depth, max_high=max_high)
                cloud = filter_cloud(cloud, image, calib_info)
                #cloud = transform(cloud, calib_info, sparse_type='angular_min', start=2.0)

                points.append(cloud)

            point1 = points[:7:6]
            point2 = points[1:8:6]
            point3 = points[2:9:6]
            point4 = points[3:10:6]
            point5 = points[4:11:6]
            point6 = points[5:12:6]
            
            '''point1 = points[:9:8]
            point2 = points[1:10:8]
            point3 = points[2:11:8]
            point4 = points[3:12:8]
            point5 = points[4:13:8]
            point6 = points[5:14:8]
            point7 = points[6:15:8]
            point8 = points[7:16:8]'''

            out5,out6 = aum(point3,sample_id)
            out7,out8 = aum(point4,sample_id)
            out1,out2 = aum(point1,sample_id)
            out3,out4 = aum(point2,sample_id)
            out9,out10 = aum(point5,sample_id)
            out11,out12 = aum(point6,sample_id)
            '''out13,out14 = aum(point7,sample_id)
            out15,out16 = aum(point8,sample_id)
            
            points1 = torch.stack((out1,out3,out5,out7,out9,out11,out13,out15), dim=0)
            points2 = torch.stack((out2,out4,out6,out8,out10,out12,out14,out16), dim=0)'''
            #points1 = torch.stack((out1,out3,out5,out7,out9,out11,out13,out15,out17,out19,out21,out23), dim=0)
            #points2 = torch.stack((out2,out4,out6,out8,out10,out12,out14,out16,out18,out20,out22,out24), dim=0)
            points1 = torch.stack((out1,out3,out5,out7,out9,out11), dim=0)
            points2 = torch.stack((out2,out4,out6,out8,out10,out12), dim=0)
            #points1 = torch.stack((out1,out3), dim=0)
            #points2 = torch.stack((out2,out4), dim=0)
        return loss, points1, points2
        #return loss, out1.unsqueeze(0), out2.unsqueeze(0)

    def save_checkpoint(self, epoch, is_best=False, filename='checkpoint.pth.tar'):
        save_dir = os.path.join(self.result_dir, 'ckpt')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_RMSE = 0  # TODO: Add RMSE loss
        state = {
            'epoch': epoch + 1,
            'arch': 'stackhourglass',
            'state_dict': self.net.state_dict(),
            'best_RMSE': best_RMSE,
            'scheduler': self.scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, save_dir + '/' + filename)
        if is_best:
            shutil.copyfile(save_dir + '/' + filename,
                            save_dir + '/model_best.pth.tar')

        #shutil.copyfile(save_dir + '/' + filename, save_dir +
        #                '/checkpoint_{}.pth.tar'.format(epoch+1))

    def strip_prefix(self, state_dict, prefix='module.'):
        if not all(key.startswith(prefix) for key in state_dict.keys()):
            return state_dict
        stripped_state_dict = {}
        for key in list(state_dict.keys()):
            stripped_state_dict[key.replace(prefix, '')] = state_dict.pop(key)
        return stripped_state_dict


def depth_to_pcl(calib, depth, max_high=1.):
    rows, cols = depth.shape
    c, r = torch.meshgrid(torch.arange(0., cols, device='cuda'),
                          torch.arange(0., rows, device='cuda'))
    points = torch.stack([c.t(), r.t(), depth], dim=0)
    points = points.reshape((3, -1))
    points = points.t()
    cloud = calib.img_to_lidar(points[:, 0], points[:, 1], points[:, 2])
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    lidar = cloud[valid]

    # pad 1 in the intensity dimension
    lidar = torch.cat(
        [lidar, torch.ones((lidar.shape[0], 1), device='cuda')], 1)
    lidar = lidar.float()
    return lidar


def transform(points, calib_info, sparse_type, start=2.):
    if sparse_type == 'angular':
        points = random_sparse_angular(points)
    if sparse_type == 'angular_min':
        points = nearest_sparse_angular(points, start)
    if sparse_type == 'angular_numpy':
        points = points.cpu().numpy()
        
        points = pto_ang_map(points).astype(np.float32)
        points = torch.from_numpy(points).cuda()

    return points

def filter_cloud(velo_points, image, calib):
    W, H = image.size
    _, _, valid_inds_fov = get_lidar_in_image_fov(
        velo_points[:, :3], calib, 0, 0, W, H, True)
    velo_points = velo_points[valid_inds_fov]
    #velo_points[:,1]=velo_points[:,1]*(-1)
    #velo_points[:,2]=velo_points[:,2]*(-1)
    # depth, width, height
    valid_inds = (velo_points[:, 0] < 35) & \
                 (velo_points[:, 0] >= 0) & \
                 (velo_points[:, 1] < 12) & \
                 (velo_points[:, 1] >= -12) & \
                 (velo_points[:, 2] < 0.5) & \
                 (velo_points[:, 2] >= -1.4)
    velo_points = velo_points[valid_inds]
    '''valid_inds1 = (velo_points[:, 0] < 12) & \
                 (velo_points[:, 1] < 3.0) & \
                 (velo_points[:, 1] >= -3.0) & \
                 (velo_points[:, 2] < 1.0) & \
                 (velo_points[:, 2] >= 0.1)
    valid_inds1 = valid_inds1==0
    velo_points = velo_points[valid_inds1]'''
    pc_data = velo_points[:,:3].cpu().numpy()
    data = FR.FliterR(pc_data)
    data,index = data.fliter_radius(pc_data)
    velo_points = velo_points[index]
    return velo_points

def gen_ang_map(velo_points, start=2., H=64, W=512, device='cuda'):
    dtheta = math.radians(0.4 * 64.0 / H)
    dphi = math.radians(90.0 / W)

    x, y, z, i = velo_points[:, 0], velo_points[:,
                                    1], velo_points[:, 2], velo_points[:, 3]

    d = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = torch.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = math.radians(45.) - torch.asin(y / r)
    phi_ = (phi / dphi).long()
    phi_ = torch.clamp(phi_, 0, W - 1)

    theta = math.radians(start) - torch.asin(z / d)
    theta_ = (theta / dtheta).long()
    theta_ = torch.clamp(theta_, 0, H - 1)
    return [theta_, phi_]


def random_sparse_angular(velo_points, H=64, W=512, slice=1, device='cuda'):
    """
    :param velo_points: Pointcloud of size [N, 4]
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    with torch.no_grad():
        theta_, phi_ = gen_ang_map(velo_points, H=64, W=512, device=device)

    depth_map = - torch.ones((H, W, 4), device=device)

    depth_map = depth_map
    velo_points = velo_points
    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]
    theta_, phi_ = theta_, phi_

    # Currently, does random subsample (maybe keep the points with min distance)
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map[theta_, phi_, 3] = i
    depth_map = depth_map.cuda()

    depth_map = depth_map[0:: slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    return depth_map[depth_map[:, 0] != -1.0]



def pto_ang_map(velo_points, H=64, W=512, slice=1):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

#   np.random.shuffle(velo_points)
    dtheta = np.radians(0.4 * 3.0 / H)
    dphi = np.radians(90.0 / W)

    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta = np.radians(2.) - np.arcsin(z / d)
    theta_ = (theta / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = - np.ones((H, W, 4))
    depth_map[theta_, phi_] = velo_points

    depth_map = depth_map[0::slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map


def nearest_sparse_angular(velo_points, start=2., H=64, W=512, slice=1, device='cuda'):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    with torch.no_grad():
        theta_, phi_ = gen_ang_map(velo_points, start, H, W, device=device)

    depth_map = - torch.ones((H, W, 4), device=device)
    depth_map = min_dist_subsample(velo_points, theta_, phi_, H, W, device='cuda')
    # depth_map = depth_map[0::slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    sparse_points = depth_map[depth_map[:, 0] != -1.0]
    return sparse_points


def min_dist_subsample(velo_points, theta_, phi_, H, W, device='cuda'):
    N = velo_points.shape[0]

    idx = theta_ * W + phi_  # phi_ in range [0, W-1]
    depth = torch.arange(0, N, device='cuda')

    sampled_depth, argmin = scatter_max(depth, idx)
    mask = argmin[argmin != -1]
    return velo_points[mask]


def save_pcl(point_cloud, path='point'):
    point_cloud = point_cloud.detach().cpu()
    np.save(path, point_cloud)


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d, pts_rect_depth = calib.lidar_to_img(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
        (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo
        

def pixel2xyz(depth, P_rect, px=None, py=None):
    
    P_rect=torch.from_numpy(P_rect).cuda()
    assert P_rect[0,1] == 0
    assert P_rect[1,0] == 0
    assert P_rect[2,0] == 0
    assert P_rect[2,1] == 0
    assert P_rect[0,0] == P_rect[1,1]
    focal_length_pixel = P_rect[0,0]
    height, width = depth.shape[:2]
    if px is None:
        px = np.tile(np.arange(width, dtype=np.float32)[None, :], (height, 1))
    if py is None:
        py = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))
    const_x = P_rect[0,2] * depth + P_rect[0,3]
    const_y = P_rect[1,2] * depth + P_rect[1,3]
   
    px=torch.from_numpy(px).cuda()
    py=torch.from_numpy(py).cuda()
    #focal_length_pixel = torch.from_numpy(focal_length_pixel)
    x = ((px * (depth + P_rect[2,3]) - const_x) / focal_length_pixel) [:, :, None]
    y = ((py * (depth + P_rect[2,3]) - const_y) / focal_length_pixel) [:, :, None]
    pc = torch.cat((x, y, depth[:, :, None]), dim=-1)
    pc[..., :2] *= -1.
    pc = pc.reshape(-1,3)
    #pc = pc.reshape(len(pc),1242,376,3)
    valid_inds = (pc[:, 0] < 20) & \
                 (pc[:, 0] >= -20) & \
                 (pc[:, 1] < 1.5) & \
                 (pc[:, 1] >= -1.4) & \
                 (pc[:, 2] < 35) & \
                 (pc[:, 2] >= 0.0)
    pc = pc[valid_inds]
    return pc
        
def aum(point,sample_id):
    pc1 = point[0]
    pc2 = point[1]
    len1 = len(pc1)
    len2 = len(pc2)  
    if len1 < len2:
        indx = len1
    else:
        indx = len2
    idx = torch.randperm(indx)
    pc1 = pc1[idx]
    pc2 = pc2[idx]
    #save_pcl(pc2, 'points/nosparse_{}'.format(sample_id))
    '''or1 = pc1[:,2] > -1.40
    or2 = pc2[:,2] > -1.40
    orr = or1 & or2
    #orr = ~orr
    pc1 = pc1[orr]
    pc2 = pc2[orr]
    
    or1 = pc1[:,0] < 35
    or2 = pc2[:,0] < 35
    orr = or1 & or2
    #orr = ~orr
    pc1 = pc1[orr]
    pc2 = pc2[orr]'''
    
    '''pc1 = point[0]
    pc2 = point[1]
    len1 = len(pc1)
    len2 = len(pc2)  
    if len1 < len2:
        indx = len1
    else:
        indx = len2
    idx = torch.randperm(indx)
    pc1 = pc1[idx]
    pc2 = pc2[idx]
    or1 = pc1[:,2] > -1.40
    or2 = pc2[:,2] > -1.40
    orr = or1 & or2
    #orr = ~orr
    pc1 = pc1[orr]
    pc2 = pc2[orr]

    orh1 = pc1[:,2] < 0.5
    orh2 = pc2[:,2] < 0.5
    orrh = orh1 & orh2
    pc1 = pc1[orrh]
    pc2 = pc2[orrh]
    
    ord1 = pc1[:,0] < 35
    ord2 = pc2[:,0] < 35
    orrd = ord1 & ord2
    #orrd = ~orrd
    pc1 = pc1[orrd]
    pc2 = pc2[orrd]    
    
    orw1 = (pc1[:,1] > -13)&(pc1[:,1] < 13)
    orw2 = (pc2[:,1] > -13)&(pc2[:,1] < 13)
    pc1 = pc1[orw1]
    pc2 = pc2[orw2]
    pc1 = pc1[:,[1,2,0]]
    pc2 = pc2[:,[1,2,0]]
    
    len1 = len(pc1)
    len2 = len(pc2)  
    if len1 < len2:
        indx = len1
    else:
        indx = len2
    idx = torch.randperm(indx)
    pc1 = pc1[idx]
    pc2 = pc2[idx]'''
    pc1 = pc1[:,[1,2,0]]
    pc2 = pc2[:,[1,2,0]]

    scale = np.diag(np.random.uniform(0.95, 1.05, 3).astype(np.float32))
    
    # rotation
    angle = np.random.uniform(-0.1745329252,0.1745329252)
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rot_matrix = np.array([[cosval, 0, sinval],
                           [0, 1, 0],
                            [-sinval, 0, cosval]], dtype=np.float32)
    matrix = scale.dot(rot_matrix.T)

    # shift
    shifts = np.random.uniform(-1,1,(1,3)).astype(np.float32)
                
    # jitter   torch.mm(bb,aa)
    #np.clip(a, a_min, a_max, out=None)
    jitter = np.clip(0.01 * np.random.randn(pc1.shape[0], 3), 0.00, 0.00).astype(np.float32)
    bias = shifts + jitter
    bias = torch.from_numpy(bias).cuda()
    matrix = torch.from_numpy(matrix).cuda()
    
    pc1_med1 = pc1[:, :3]
    pc1_med2 = torch.mm(pc1_med1,matrix)
    pc1_med3 = torch.add(pc1_med2,bias)
    #pc1_med4 = torch.cat([pc1_med3, torch.ones((pc1.shape[0], 1), device='cuda')], 1)
    pc2_med1 = pc2[:, :3]
    pc2_med2 = torch.mm(pc2_med1,matrix)
    pc2_med3 = torch.add(pc2_med2,bias)
    # pc2, order: rotation, shift, jitter
    # rotation
    angle2 = np.random.uniform(0.0 ,0.0)
    cosval2 = np.cos(angle2)
    sinval2 = np.sin(angle2)
    matrix2 = np.array([[cosval2, 0, sinval2],
                        [0, 1, 0],
                        [-sinval2, 0, cosval2]], dtype=np.float32)
    matrix2 = torch.from_numpy(matrix2.T).cuda()
    # shift
    shifts2 = np.random.uniform(-0.3,
                                0.3,
                                (1, 3)).astype(np.float32)

    pc2_med4 = torch.mm(pc2_med3,matrix2)
    shifts2 = torch.from_numpy(shifts2).cuda()
    pc2_med5 = torch.add(pc2_med4,shifts2)
    #sf = pc2[:, :3] - pc1[:, :3]
    idx2 = torch.randperm(len(pc2_med5))
    idx1 = torch.randperm(len(pc1_med3))
    indxx1 = idx1[:4096]
    indxx2 = idx2[:4096]
    out3 = pc1_med3[indxx1]
    out4 = pc2_med5[indxx2]
    return out3, out4
    
def removep(points,sample_id):
    pc1 = points[0]
    pc2 = points[1]
    '''len1 = len(pc1)
    len2 = len(pc2)  
    if len1 < len2:
        indx = len1
    else:
        indx = len2
    idx = torch.randperm(indx)
    pc1 = pc1[idx]
    pc2 = pc2[idx]
    pc11 = pc1[:,[1,2,0]]
    pc22 = pc2[:,[1,2,0]]
    
    idx2 = torch.randperm(len(pc22))
    idx1 = torch.randperm(len(pc11))
    indxx1 = idx1[:8192]
    indxx2 = idx2[:8192]
    out1 = pc11[indxx1]
    sf = sf[indxx1]
    out2 = pc22[indxx2]'''
    
    len1 = len(pc1)
    len2 = len(pc2)  
    if len1 < len2:
        indx = len1
    else:
        indx = len2
    idx = torch.randperm(indx)
    pc1 = pc1[idx]
    pc2 = pc2[idx]

    pc1 = pc1[:,[1,2,0]]
    pc2 = pc2[:,[1,2,0]]
    
    scale = np.diag(np.random.uniform(0.95, 1.05, 3).astype(np.float32))
    # rotation
    angle = np.random.uniform(-0.1745329252,0.1745329252)
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rot_matrix = np.array([[cosval, 0, sinval],
                           [0, 1, 0],
                            [-sinval, 0, cosval]], dtype=np.float32)
    matrix = scale.dot(rot_matrix.T)

    # shift
    shifts = np.random.uniform(-1.0,1.0,(1,3)).astype(np.float32)
    # jitter   torch.mm(bb,aa)
    #np.clip(a, a_min, a_max, out=None)
    jitter = np.clip(0.01 * np.random.randn(pc1.shape[0], 3), 0.00, 0.00).astype(np.float32)
    bias = shifts + jitter
    bias = torch.from_numpy(bias).cuda()
    matrix = torch.from_numpy(matrix).cuda()
    pc1_med1 = pc1[:, :3]
    pc1_med2 = torch.mm(pc1_med1,matrix)
    pc1_med3 = torch.add(pc1_med2,bias)
    #pc1_med4 = torch.cat([pc1_med3, torch.ones((pc1.shape[0], 1), device='cuda')], 1)
    pc2_med1 = pc2[:, :3]
    pc2_med2 = torch.mm(pc2_med1,matrix)
    pc2_med3 = torch.add(pc2_med2,bias)
    # pc2, order: rotation, shift, jitter
    # rotation
    angle2 = np.random.uniform(0.0 ,0.0)
    cosval2 = np.cos(angle2)
    sinval2 = np.sin(angle2)
    matrix2 = np.array([[cosval2, 0, sinval2],
                        [0, 1, 0],
                        [-sinval2, 0, cosval2]], dtype=np.float32)
    matrix2 = torch.from_numpy(matrix2.T).cuda()
    # shift
    shifts2 = np.random.uniform(-0.3,
                                0.3,
                                (1, 3)).astype(np.float32)

    pc2_med4 = torch.mm(pc2_med3,matrix2)
    shifts2 = torch.from_numpy(shifts2).cuda()
    pc2_med5 = torch.add(pc2_med4,shifts2)
    #sf = pc2[:, :3] - pc1[:, :3]
    idx2 = torch.randperm(len(pc2_med5))
    idx1 = torch.randperm(len(pc1_med3))
    indxx1 = idx1[:8192]
    indxx2 = idx2[:8192]
    out1 = pc1_med3[indxx1]
    out2 = pc2_med5[indxx2]
    #out1 = pc1[indxx1]
    #out2 = pc2[indxx2]
    #sf = sf[indxx1]
    return out1, out2
