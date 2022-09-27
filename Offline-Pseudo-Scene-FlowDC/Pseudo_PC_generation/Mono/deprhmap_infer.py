import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import model_io
import utils
from models import UnetAdaptiveBins
# ---------------------------------------  02 ---------------------------------------
# selfcu, selfcv, selffu, selffv, selftx, selfty = 607.1928,185.2157, 718.8560,718.8560,-0.0631,0.0002
# selfR0 = torch.tensor([[ 0.9999,  0.0073, -0.0075],
#     [-0.0073,  1.0000, -0.0044],
#     [ 0.0075,  0.0044,  1.0000]]).cuda()
# selfC2V_t = torch.tensor([[ 7.9675e-03, -9.9997e-01, -8.4623e-04],
#     [-2.7711e-03,  8.2417e-04, -1.0000e+00],
#     [ 9.9996e-01,  7.9698e-03, -2.7644e-03],
#     [ 2.9180e-01, -1.1406e-02, -5.6239e-02]]).cuda()

# ---------------------------------------  00 ---------------------------------------

selfcu, selfcv, selffu, selffv, selftx, selfty = 607.1928,185.2157, 718.8560,718.8560,-0.0631,0.0002
selfR0 = torch.tensor([[ 0.9999,  0.0073, -0.0075],
    [-0.0073,  1.0000, -0.0044],
    [ 0.0075,  0.0044,  1.0000]]).cuda()
selfC2V_t = torch.tensor([[ 7.9675e-03, -9.9997e-01, -8.4623e-04],
    [-2.7711e-03,  8.2417e-04, -1.0000e+00],
    [ 9.9996e-01,  7.9698e-03, -2.7644e-03],
    [ 2.9180e-01, -1.1406e-02, -5.6239e-02]]).cuda()

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, target_size=(640, 480)):
        # image = image.resize(target_size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class InferenceHelper:
    def __init__(self, dataset='kitti', device='cuda:0'):
        self.toTensor = ToTensor()
        self.device = device
        if dataset == 'nyu':
            self.min_depth = 1e-3
            self.max_depth = 10
            self.saving_factor = 1000  # used to save in 16 bit
            model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            pretrained_path = "./pretrained/AdaBins_kitti.pt"
        elif dataset == 'kitti':
            self.min_depth = 1e-3
            self.max_depth = 80
            self.saving_factor = 256
            model = UnetAdaptiveBins.build(n_bins=256, min_val=self.min_depth, max_val=self.max_depth)
            pretrained_path = "./pretrained/AdaBins_kitti.pt"
        else:
            raise ValueError("dataset can be either 'nyu' or 'kitti' but got {}".format(dataset))

        model, _, _ = model_io.load_checkpoint(pretrained_path, model)
        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def predict_pil(self, pil_image, visualized=False):
        # pil_image = pil_image.resize((640, 480))
        img = np.asarray(pil_image) / 255.

        img = self.toTensor(img).unsqueeze(0).float().to(self.device)
        bin_centers, pred = self.predict(img)

        if visualized:
            viz = utils.colorize(torch.from_numpy(pred).unsqueeze(0), vmin=None, vmax=None, cmap='magma')
            # pred = np.asarray(pred*1000, dtype='uint16')
            viz = Image.fromarray(viz)
            return bin_centers, pred, viz
        return bin_centers, pred

    @torch.no_grad()
    def predict(self, image):
        bins, pred = self.model(image)
        pred = np.clip(pred.cpu().numpy(), self.min_depth, self.max_depth)

        # Flip
        image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(self.device)
        pred_lr = self.model(image)[-1]
        pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], self.min_depth, self.max_depth)

        # Take average of original and mirror
        final = 0.5 * (pred + pred_lr)
        final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:],
                                          mode='bilinear', align_corners=True).cpu().numpy()

        final[final < self.min_depth] = self.min_depth
        final[final > self.max_depth] = self.max_depth
        final[np.isinf(final)] = self.max_depth
        final[np.isnan(final)] = self.min_depth

        centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        centers = centers.cpu().squeeze().numpy()
        centers = centers[centers > self.min_depth]
        centers = centers[centers < self.max_depth]

        return centers, final

    @torch.no_grad()
    def predict_dir(self, test_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        transform = ToTensor()
        all_files = glob.glob(os.path.join(test_dir, "*"))
        self.model.eval()
        for f in tqdm(all_files):
            image = np.asarray(Image.open(f), dtype='float32') / 255.
            image = transform(image).unsqueeze(0).to(self.device)

            centers, final = self.predict(image)
            # final = final.squeeze().cpu().numpy()

            final = (final * self.saving_factor).astype('uint16')
            basename = os.path.basename(f).split('.')[0]
            save_path = os.path.join(out_dir, basename + ".png")

            Image.fromarray(final.squeeze()).save(save_path)

def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    ones = torch.ones((pts.shape[0], 1), dtype=torch.float32).cuda()
    pts_hom = torch.cat((pts, ones), dim=1)
    return pts_hom

def img_to_rect(u, v, depth_rect):
    """
    :param u: (N)
    :param v: (N)
    :param depth_rect: (N)
    :return:
    """
    x = ((u - selfcu) * depth_rect) / selffu + selftx
    y = ((v - selfcv) * depth_rect) / selffv + selfty
    pts_rect = torch.cat(
        (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), dim=1)
    return pts_rect

def rect_to_lidar(pts_rect):
    """
    :param pts_rect: (N, 3)
    :return pts_lidar: (N, 3)
    """
    pts_hom = cart_to_hom(torch.matmul(
        pts_rect, torch.inverse(selfR0.t())))
    
    pts_rect = torch.matmul(pts_hom, selfC2V_t)
    return pts_rect

def depth_to_pcl(depth):
    max_high=1.
    rows, cols = depth.shape
    c, r = torch.meshgrid(torch.arange(0., cols, device='cuda'),
                          torch.arange(0., rows, device='cuda'))
    points = torch.stack([c.t(), r.t(), depth], dim=0)
    points = points.reshape((3, -1))
    points = points.t()
    cloud = img_to_rect(points[:, 0], points[:, 1], points[:, 2])

    cloud = rect_to_lidar(cloud)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    lidar = cloud[valid]

    # pad 1 in the intensity dimension
    lidar = torch.cat(
        [lidar, torch.ones((lidar.shape[0], 1), device='cuda')], 1)
    lidar = lidar.float()
    return lidar


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from time import time
    # start = time()
    inferHelper = InferenceHelper()
    for i in range(0, 4661):
        img = Image.open("/dataset/data_odometry_color/02/image_2/{:06d}.png".format(i))
        centers, pred = inferHelper.predict_pil(img)

        # pred = torch.tensor(pred.squeeze().squeeze()).cuda()
        # PC = depth_to_pcl(pred)
        np.save('/dataset/data_odometry_color/02/depth_map/{:06d}.npy'.format(i), pred.squeeze())
        if i % 50 == 0:
            print('+++++++++++++++++++++{:06d}++++++++++++++++'.format(i))

    # print(f"took :{time() - start}s")
    # plt.imshow(pred.squeeze(), cmap='magma_r')
    # plt.savefig('test_imgs/test{}.png'.format(102))
    #plt.show()
