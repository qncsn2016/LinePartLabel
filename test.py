import json
import numpy as np
from easydict import EasyDict as edict

a=json.load(open("train2017_annotations/instances_train2017.json",'r'))
ea=edict(a)

carimgs = []
catimgs = []

for an in ea.annotations:
    if an['category_id'] == 2:
        carimgs.append(an['image_id'])
    if an['category_id'] == 18:
        catimgs.append(an['image_id'])

carimgs=list(set(carimgs))
catimgs=list(set(catimgs))

carfilenames = []
catfilenames = []
for img in ea.images:
    if img.id in carimgs:
        carfilenames.append(img.file_name)
    elif img.id in catimgs:
        catfilenames.append(img.file_name)

np.savetxt("bicyclefilenames2.txt",carfilenames,fmt="%s",delimiter='\n')
np.savetxt("dogfilenames2.txt",catfilenames,fmt="%s",delimiter='\n')

# 复制文件
from shutil import copyfile
copyfile(src, dst)
copyfile('./test.txt', '/home/gaosiqi/tmp/test.txt')

dog1=list(np.loadtxt("dogfilenames1.txt",dtype=str,delimiter='\n'))
while len(dog1)>0:
    c=dog1.pop()
    copyfile('val2017/'+c,'mydog/val/'+c)

mycat/val       173   163   163
mycat/train     3950  3808  3808
mycar/val       534   472   472
mycar/train     12251 11094 11904

mydog/val       168   146   146
mydog/train     4245  3919  3919
mybicycle/val   149   119   118
mybicycle/train 3252  2164  2164