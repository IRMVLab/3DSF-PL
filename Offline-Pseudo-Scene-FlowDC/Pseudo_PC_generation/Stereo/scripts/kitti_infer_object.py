from PIL import Image
import open3d
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../') # add relative path

from module.sttr import STTR
from dataset.preprocess import normalization, compute_left_occ_region
from utilities.misc import NestedTensor
import utilities.calibration as calibration

# Default parameters
args = type('', (), {})() # create empty args
args.channel_dim = 128
args.position_encoding='sine1d_rel'
args.num_attn_layers=6
args.nheads=8
args.regression_head='ot'
args.context_adjustment_layer='cal'
args.cal_num_blocks=8
args.cal_feat_dim=16
args.cal_expansion_ratio=4

model = STTR(args).cuda().eval()

model_file_name = "../pretrain_weights/kitti_finetuned_model.pth.tar"
checkpoint = torch.load(model_file_name)
pretrained_dict = checkpoint['state_dict']
model.load_state_dict(pretrained_dict, strict=False)
print("Pre-trained model successfully loaded.")
disp = None

# i = 620
# if i == 620:
for i in range(0, 7481):
    calib_file = '/dataset/KITTI/object/training/calib/{:06d}.txt'.format(i)
    left = np.array(Image.open("/dataset/KITTI/object/training/image_2/{:06d}.png".format(i)))
    right = np.array(Image.open("/dataset/KITTI/object/training/image_3/{:06d}.png".format(i)))
    # disp = np.array(Image.open('../sample_data/KITTI_2015/training/disp_occ_0/000046_10.png')).astype(np.float) / 256.
    input_data = {'left': left, 'right':right, 'disp':disp}
    input_data = normalization(**input_data)

    h, w, _ = left.shape
    bs = 1

    downsample = 3
    col_offset = int(downsample / 2)
    row_offset = int(downsample / 2)
    sampled_cols = torch.arange(col_offset, w, downsample)[None,].expand(bs, -1).cuda()
    sampled_rows = torch.arange(row_offset, h, downsample)[None,].expand(bs, -1).cuda()
    with torch.no_grad():
        # build NestedTensor
        input_data = NestedTensor(input_data['left'].cuda()[None,],input_data['right'].cuda()[None,], sampled_cols=sampled_cols, sampled_rows=sampled_rows)
        output = model(input_data)

        disp_pred = output['disp_pred'].data.cpu().numpy()[0]
        occ_pred = output['occ_pred'].data.cpu().numpy()[0] > 0.5
        disp_pred[occ_pred] = 1000000.0
        
        calib = calibration.Calibration(calib_file)
        lidar_pc = calib.depthmap_to_lidar(disp_pred)
        #depthmap = calib.disp_to_depth(disp_pred)

        np.save('/dataset/KITTI/object/training/Stereo_pc/{:06d}.npy'.format(i), lidar_pc)
        # torch.cuda.empty_cache()
        if i % 50 == 0:
            print('+++++++++++++++++++++{:06d}++++++++++++++++'.format(i))
        
    # plt.imshow(disp_pred, cmap='magma_r')
    # plt.savefig('test{}.png'.format(620))

# print('--',disp_pred,disp_pred.shape)