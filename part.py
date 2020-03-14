import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from utils import apply_mask,random_colors
import runthis

def linepart(img, mask, roi, line_rgbrange):
    """
    按分隔线和目标级别mask划分区域
    :param img: 图像
    :param mask: 目标级别mask 
    :param roi: list ROI
    :param line_rgbrange: list 分隔线的RGB范围 
    :return: 
    """
    img = img
    mask = mask
    minrow = roi[0]
    mincol = roi[1]
    maxrow = roi[2]
    maxcol = roi[3]

    def rightpix(row, col):
        """
        判断这个点是否符合膨胀条件
        :param row: 点的行
        :param col: 点的列
        :return: boolean
        """
        if row <0 or row >=img.shape[0] or col<0 or col>=img.shape[1]:
            return False
        # if row <= minrow or row >= maxrow or col <= mincol or col >= maxcol:
        if row < minrow or row > maxrow or col < mincol or col > maxcol:
            return False
        if mask[row][col] != 1:
            return False
        rgb = img[row][col]
        if line_rgbrange[0] <= rgb[0] <= line_rgbrange[1] \
                and line_rgbrange[2] <= rgb[1] <= line_rgbrange[3] \
                and line_rgbrange[4] <= rgb[2] <= line_rgbrange[5]:
            return False
        return True

    def dilate(row, col, labelcount):
        """
        每个部件的膨胀算法
        :param row: 种子的行
        :param col: 种子的列
        :param labelcount: 种子所在区域打标签几
        :return: 种子所在区域打标签得到的mask
        """
        queue = [(row, col)]
        around = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        part_mask = np.zeros(mask.shape)
        inqueued = np.zeros(mask.shape)
        inqueued[row][col] = 1
        while len(queue) > 0:
            p = queue.pop()
            if not rightpix(p[0], p[1]):
                continue
            part_mask[p[0]][p[1]] = labelcount
            mask[p[0]][p[1]]=2
            for i in range(len(around)):
                newrow = p[0] + around[i][0]
                newcol = p[1] + around[i][1]
                if rightpix(newrow, newcol) and inqueued[newrow][newcol] == 0:
                    queue.append((newrow, newcol))
                    inqueued[newrow][newcol] = 1
        return part_mask

    def finishpart():
        # 判断分割是否结束
        unlabeled=np.where(mask==1)
        for i in range(len(unlabeled[0])):
            rgb = img[unlabeled[0][i]][unlabeled[1][i]]
            if not(line_rgbrange[0] <= rgb[0] <= line_rgbrange[1] \
                    and line_rgbrange[2] <= rgb[1] <= line_rgbrange[3] \
                    and line_rgbrange[4] <= rgb[2] <= line_rgbrange[5]):
                return False, unlabeled[0][i],unlabeled[1][i]  # 如果有不是线而且mask为1的 说明还没part完
        return True,0,0

    labelbase = runthis.get_labelbase()
    labelcount=1
    all_part_mask=[]
    while True:
        isfinish, wrongrow, wrongcol=finishpart()
        if isfinish:
            break
        part_mask = dilate(wrongrow, wrongcol, labelbase * 10 + labelcount)
        all_part_mask.append(part_mask)
        labelcount += 1
    # while not finishpart():
    #     while not rightpix(seed_row, seed_col):
    #         seed_row = random.randint(minrow, maxrow)
    #         seed_col = random.randint(mincol, maxcol)
    #     # print("seed_row =", seed_row, "  seed_col =", seed_col)
    #     part_mask = dilate(seed_row, seed_col, labelbase*10+labelcount)
    #     all_part_mask.append(part_mask)
    #     labelcount += 1
    n_parts=len(all_part_mask)
    colors=random_colors(n_parts)
    masked_image=img
    for i in range(n_parts):
        masked_image = apply_mask(masked_image, all_part_mask[i], colors[i])
        mask=np.where(all_part_mask[i]>0,all_part_mask[i],mask)

    return all_part_mask, mask, masked_image
