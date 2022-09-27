import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
# for i in range(10000):
#     os.path.dirname('D:\\newfile/{:6}'.format(i)+'.npy')
path_dir = '/dataset/data_odometry_color/02/Stereo_depth/000010.npy'
output_directory = os.path.dirname(path_dir)  # 提取文件的路径
output_name = os.path.splitext(os.path.basename(path_dir[-8:-3]))  # 提取文件名
arr = np.load(path_dir)# .astype(np.float64)  # 提取 npy 文件中的数组
print(arr,arr.shape)
disp_to_img = scipy.misc.imresize( arr , arr.shape)  # 根据 需要的尺寸进行修改

plt.imshow(disp_to_img, cmap='plasma')
plt.savefig('000.png')
# plt.imsave('000.png', disp_to_img, cmap='viridis')  # 定义命名规则，保存图片
