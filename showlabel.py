import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import cv2
import runthis

from utils import apply_mask, random_colors

if __name__ == '__main__':
    del_bg_dir=runthis.get_del_bg_dir()
    label_dir=runthis.get_label_dir()
    rois_dir = runthis.get_rois_dir()

    img_paths = next(os.walk(label_dir))[2]
    colors=random_colors(10)
    for img_path in img_paths:
        img = cv2.imread(del_bg_dir + img_path.split('.')[0]+'_delbg.png')     # 反常
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 正常
        rois = np.loadtxt(rois_dir + img_path.split('.')[0] + '.rois', dtype=np.int, delimiter=',').tolist()
        mask_inone=np.loadtxt(label_dir+img_path,dtype=int,delimiter=',')
        minrow = rois[0]
        mincol = rois[1]
        maxrow = rois[2]
        maxcol = rois[3]

        temp1=mask_inone.flatten().tolist()
        labels = list(set(temp1))
        labels.sort()
        labels=labels[1:]
        mask_part=[]
        for i in range(len(labels)):
            tempmask=np.zeros(mask_inone.shape,dtype=int)
            tempmask=np.where(mask_inone==labels[i],np.ones(tempmask.shape, dtype=int) * labels[i],tempmask)
            img=apply_mask(img,tempmask,colors[labels[i]%10])
        plt.figure(figsize=(10,5))
        ax=plt.subplot(1,2,1)
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])
        ax=plt.subplot(1,2,2)
        plt.imshow(img[minrow:maxrow,mincol:maxcol:])
        plt.xticks([]), plt.yticks([])
        plt.show()