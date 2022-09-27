# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:52:47 2021

@author: jck
"""
import numpy as np
pts_file = '/home/jck/SfMLearner-master/output2/0000000000.bin'
aug_pts = np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)
np.save('/home/jck/SfMLearner-master/output2/0',aug_pts)