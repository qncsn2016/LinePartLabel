#! /usr/bin/env python
# coding=utf-8
#================================================================
#
#   Editor      : PyCharm
#   File name   : anchor2graph.py
#   Author      : Jingyi Wang
#   Created date: 2020/3/12 11:17
#   Description : get the graph of a class from an anchor image
#
#================================================================
import cv2
import numpy as np

# img_path='D:\\MyProjects\\Datasets\\COCO2017\\mybicycle\\trainsmallExp\\image_line\\000000014750_line.png'


d={2:'bicycle',16:'cat',3:'car',17:'dog'}
# for bicycle
# labelbase = 2
# imageid='000000014750'
# mask_path='D:\\MyProjects\\Datasets\\COCO2017\\my'+d[labelbase] + '\\trainsmallExp\\mini_label_part\\' + imageid + '.png'
# for dog
labelbase = 17
imageid='000000273232'
labelline_path='D:\\MyProjects\\Datasets\\COCO2017\\my'+d[labelbase] + '\\valsmallExp\\label_beflabelit\\' + imageid + '.txt'
mask_path='D:\\MyProjects\\Datasets\\COCO2017\\my'+d[labelbase] + '\\valsmallExp\\mini_label_part\\' + imageid + '.png'
goes=[[-1,-1],[0,-1],[1,-1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]

# 处理人工label之后的 不带line的
mask=cv2.imread(mask_path)[:,:,:1]
mask=np.squeeze(mask)
h, w = mask.shape
labels=np.array(list(set(mask.flatten().tolist())))
labels=labels[labels>=labelbase]
n_part=len(labels)

# TODO 如果真有恰好竖直的线得考虑这部分注释掉的代码
# 人工label之前的 带有line
# labelline=np.loadtxt(labelline_path,dtype=int,delimiter=',')
# rows,cols=np.where(labelline==1)
# for i in range(len(rows)):
#     row, col = rows[i], cols[i]
#     for j in range(len(goes)):
#         row2=row+goes[j][0]
#         col2=col+goes[j][1]
#         if row2 < 0 or row2 >= h or col2 < 0 or col2 >= w:
#             continue
#         if labelline[row2][col2] > labelbase:
#             mask[row][col]=labelline[row2][col2]
#             break


graph=np.zeros((n_part+1,n_part+1))
for label in labels:
    rows,cols=np.where(mask==label)
    for i in range(len(rows)):
        row, col = rows[i], cols[i]
        for j in range(len(goes)):
            row2=row+goes[j][0]
            col2=col+goes[j][1]
            if row2<0 or row2>=h or col2<0 or col2>=w:
                continue
            if mask[row2][col2]==0 or mask[row2][col2]==label:
                continue
            graph[label%10][mask[row2][col2]%10]=1
            graph[mask[row2][col2]%10][label%10]=1
print(graph)
np.savetxt('graph_'+d[labelbase]+'.txt', graph, fmt='%d',delimiter=',')

