import cv2
import numpy as np
import os
import re
import runthis

line_part_dir=runthis.get_line_part_dir()
paths=next(os.walk(line_part_dir))[2]

print('打开image_line_part中的图片')
while True:
    path_part=input('输入要纠正的图片序号(不用输前缀0和后缀类型)：')
    try:
        if path_part=='.':
            print('退出')
            break
        elif path_part=='r':
            continue
        for p in paths:
            if re.search(path_part,p):
                path=p
                break
        n_part=input('输入实际有几个部分：')

        # img=cv2.imread('image_line_part\\'+path)
        # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        mask=np.loadtxt('label_beflabelit\\'+path.split('_')[0]+'.txt',dtype=int,delimiter=',')
        f=mask.flatten()
        l=f.tolist()

        d={}
        for key in l:
            if key==0 or key==1:
                continue
            d[key]=d.get(key,0)+1
        pairs=list(d.items())
        pairs.sort(key=lambda x:x[1])
        wrongpairs=pairs[:len(pairs)-int(n_part)]
        if len(wrongpairs)==0:
            continue
        wronglabels=[]
        for p in wrongpairs:
            ps=np.ones(mask.shape)*p[0]
            mask=np.where(mask==ps,0,mask)
        np.savetxt('label_beflabelit\\'+path.split('_')[0]+'.txt',mask,fmt='%d',delimiter=',')
    except Exception as e:
        print(path_part,'出错',e)
        continue
