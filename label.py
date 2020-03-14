import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import cv2
import runthis

from utils import apply_mask, random_colors

plt.rcParams['figure.figsize'] = (15, 5)

def labelit(img, mask_part, labelbase, rois):
    """
    人工给每部分重新打标签
    :param img: 图片
    :param mask_part: 标签
    :return: 重打过后的mask
    """
    n_part = len(mask_part)
    colors = random_colors(n_part)
    minrow = rois[0]
    mincol = rois[1]
    maxrow = rois[2]
    maxcol = rois[3]
    newmask = np.zeros(mask_part[0].shape)
    newmask = newmask.astype(np.int32)
    print("按从左到右顺序输入标签:")

    for i in range(n_part):
        mask = mask_part[i]
        temp = copy.copy(img)
        onepart = apply_mask(temp, mask, colors[i])
        onepart = onepart[minrow:maxrow, mincol:maxcol, ]

        # print(i,onepart.shape)
        ax=plt.subplot(1, n_part, i + 1)
        # print(ax)
        plt.imshow(onepart)
        ax.set_title('part')
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0, 0)
    plt.show()

    labels = str(input())
    if labels[0]=='.':
        return []
    # if labels[0]=='r' or labels[0]=='R':
    #     for i in range(n_part):
    #         mask = mask_part[i]
    #         temp = copy.copy(img)
    #         onepart = apply_mask(temp, mask, colors[i])
    #         onepart = onepart[minrow:maxrow, mincol:maxcol, ]
    #         cv2.imwrite(str(i)+'.png',onepart)
    #     for i in range(n_part):
    #         onepart=cv2.imread(str(i)+'.png')
    #         ax = plt.subplot(1, n_part, i + 1)
    #         plt.imshow(onepart)
    #         ax.set_title('part')
    #         plt.xticks([]), plt.yticks([])
    #         plt.tight_layout()
    #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #     plt.margins(0, 0)
    #     plt.show()
    for i in range(n_part):
        if int(labels[i])==0:
            newmask = np.where(mask_part[i] > 0,
                               np.zeros(newmask.shape),
                               newmask)  # 忽然发现这句好像没用?
        else:
            newmask = np.where(mask_part[i] > 0,
                               np.ones(newmask.shape, dtype=int) * (labelbase*10+int(labels[i])),
                               newmask)

    # 一部分一部分标
    # for mask in mask_part:
    #     plt.figure(figsize=(10, 10))
    #     print("输入这部分的标签:")
    #     temp=copy.copy(img)
    #     onepart=apply_mask(temp,mask,colors[0])
    #     plt.subplot(1,1,1)
    #     plt.imshow(onepart)
    #     plt.title("输入这部分的标签")
    #     plt.xticks([]), plt.yticks([])
    #     # plt.show()
    #     plt.ion()
    #     plt.pause(2)
    #     plt.close()
    #     newlabel=int(input())
    #     newmask=np.where(mask>0,newlabel,newmask)
    return newmask

if __name__ == '__main__':
    del_bg_dir=runthis.get_del_bg_dir()
    line_dir=runthis.get_line_dir()
    label_dir=runthis.get_label_dir()
    label_beflabelit_dir = runthis.get_label_beflabelit_dir()
    rois_dir=runthis.get_rois_dir()
    labelbase=runthis.get_labelbase()

    labeled=next(os.walk(label_dir))[2]
    labeled.sort()
    img_paths = next(os.walk(line_dir))[2]
    img_paths.sort()
    if len(labeled)>0:
        img_paths=img_paths[img_paths.index(labeled[-1].split('.')[0]+'_line.png'):]
    for img_path in img_paths:
        img = cv2.imread(del_bg_dir + img_path.split('.')[0][:-5]+'_delbg.png')     # 反常
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                  # 正常
        rois = np.loadtxt(rois_dir + img_path.split('.')[0][:-5] + '.rois', dtype=np.int, delimiter=',').tolist()

        mask_inone=np.loadtxt(label_beflabelit_dir+img_path.split('.')[0][:-5]+'.txt',dtype=int,delimiter=',')
        temp1=mask_inone.flatten().tolist()
        oldlabels = list(set(temp1))
        oldlabels.sort()
        oldlabels = oldlabels[2:]
        mask_part=[]
        for i in oldlabels:
            tempmask=np.zeros(mask_inone.shape,dtype=int)
            tempmask=np.where(mask_inone==i,np.ones(tempmask.shape, dtype=int) * i,tempmask)
            mask_part.append(tempmask)

        newmask = labelit(img, mask_part, labelbase, rois)
        plt.clf()
        writename = label_dir + img_path.split('.')[0][:-5] + ".txt"
        if len(newmask)==0:
            print("part错误 没有标记", writename)
            continue
        np.savetxt(writename, newmask, fmt="%d", delimiter=",")
        print("保存像素级部件标签", writename)
    print('处理完毕')