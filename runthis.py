import cv2
import numpy as np
import copy
import os
from line import Handwriting, lineit
from part import linepart
from label import labelit

# 在这个文件的class_name里: 3-car 16-cat 17-dog 2-bicycle 代码以这个为准
# 在COCO数据集的json里:     3-car 17-cat 18-dog 2-bicycle
d={2:'bicycle',16:'cat',3:'car',17:'dog'}
labelbase = 17  # TODO 类别
t={0:'val',1:'train'}
istrain=1  # TODO 标train的还是val的

# img_dir= 'image\\'                # 原图
del_bg_dir= 'D:\\MyProjects\\Datasets\\COCO2017\my'+d[labelbase]+'\\'+t[istrain]+'delbg3\\'       # 删了背景的
line_dir='image_line\\'           # 画了线的
line_part_dir='image_line_part\\' # 根据画线apply_mask 不同区域不同颜色的
label_dir='label_part\\'          # 部件级标签
label_beflabelit_dir='label_beflabelit\\'  # 规范前的部件级标签
mini_masks_dir='D:\\MyProjects\\Datasets\\COCO2017\my'+d[labelbase]+'\\'+t[istrain]+'minimasks\\'
rois_dir='D:\\MyProjects\\Datasets\\COCO2017\my'+d[labelbase]+'\\'+t[istrain]+'rois\\'

def get_del_bg_dir():
    return del_bg_dir
def get_line_dir():
    return line_dir
def get_line_part_dir():
    return line_part_dir
def get_label_dir():
    return label_dir
def get_label_beflabelit_dir():
    return label_beflabelit_dir
def get_mini_masks_dir():
    return mini_masks_dir
def get_rois_dir():
    return rois_dir
def get_labelbase():
    return labelbase

if __name__ == '__main__':
    img_paths=next(os.walk(del_bg_dir))[2]
    for img_path in img_paths:
        # 读入数据
        img = cv2.imread(del_bg_dir + img_path)       # 反常
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # 正常

        mask_object=np.squeeze(cv2.imread(mini_masks_dir+ img_path.split('.')[0] + '.png')[:,:,[0]])
        rois=np.loadtxt(rois_dir+img_path.split('.')[0] + '.rois',dtype=np.int,delimiter=',').tolist()

        # 画线
        img_line = lineit(copy.copy(img), img_path, rois)
        # img_line = cv2.cvtColor(img_line, cv2.COLOR_RGB2BGR)  # 反常
        # writename = line_dir + img_path.split('.')[0] + "_line.png"
        # cv2.imwrite(writename, img_line)  # 写入之后正常
        # print("保存画线图片", writename)
        # img_line = cv2.cvtColor(img_line, cv2.COLOR_BGR2RGB)  # 正常

        # 按线划分部件
        mask_part, mask_inone, img_part = linepart(copy.copy(img_line), mask_object, rois, [200, 255, 0, 100, 0, 100])
        # img_part = cv2.cvtColor(img_part, cv2.COLOR_RGB2BGR)  # 反常
        # writename = line_dir + img_path.split(".")[0] + "_part.png"
        # cv2.imwrite(writename, img_part)  # 写入之后正常
        # print("保存划分部件图片", writename)
        # img_part = cv2.cvtColor(img_part, cv2.COLOR_BGR2RGB)  # 正常

        # 结果展示
        # plt.figure(figsize=(15, 10))
        #
        # ax=plt.subplot(1, 3, 1)
        # plt.imshow(img)
        # ax.set_title("原图",fontsize=22)
        # plt.xticks([]), plt.yticks([])
        #
        # ax=plt.subplot(1, 3, 2)
        # plt.imshow(img_line)
        # ax.set_title("分隔线",fontsize=22)
        # plt.xticks([]), plt.yticks([])
        #
        # ax=plt.subplot(1, 3, 3)
        # plt.imshow(img_part)
        # ax.set_title("划分部件",fontsize=22)
        # plt.xticks([]), plt.yticks([])
        #
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.margins(0, 0)
        # plt.ion()
        # plt.pause(3)
        # plt.close()

        # 部件标注
        newmask = labelit(img, mask_part, labelbase, rois)
        writename= label_dir + img_path.split('.')[0] + ".txt"
        np.savetxt(writename,newmask,fmt="%d",delimiter=",")
        print("保存像素级部件标签", writename)