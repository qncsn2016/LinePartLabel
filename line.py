import matplotlib.pyplot as plt
import cv2
import runthis
import os
import numpy as np
import copy

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
img = None
rois=None
minrow=None
mincol=None
maxrow=None
maxcol=None

class Handwriting:
    def __init__(self, line, img_path):
        self.line = line
        self.press = None  # 状态标识，1为按下，None为没按下
        self.cidpress = line.figure.canvas.mpl_connect('button_press_event', self.on_press)  # 鼠标按下事件
        self.cidrelease = line.figure.canvas.mpl_connect('button_release_event', self.on_release)  # 鼠标放开事件
        self.cidmotion = line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)  # 鼠标拖动事件

        self.lastpix = None
        self.nowpix = None
        self.img_path=img_path

    def on_press(self, event):  # 鼠标按下调用
        if event.inaxes != self.line.axes: return
        if event.button == 3:  # 按下鼠标右键保存
            print(self.img_path,'保存')
            plt.close()
        if event.button == 2:  # 按下滑轮 该图片不好
            print(self.img_path,'舍弃')
            with open('bad_imgs.txt','a') as f:
                f.write(self.img_path+'\n')
            plt.close()
        self.press = 1
        self.lastpix = [event.xdata, event.ydata]

    def on_release(self, event):
        if event.inaxes != self.line.axes: return
        self.press = None
        self.lastpix = None

    def on_motion(self, event):  # 鼠标拖动调用
        if event.inaxes != self.line.axes: return
        if self.press is None: return
        self.nowpix = [event.xdata, event.ydata]
        plotx = [self.lastpix[0], self.nowpix[0]]
        ploty = [self.lastpix[1], self.nowpix[1]]
        plt.plot(plotx, ploty, color='r', linewidth=1)
        self.line.figure.canvas.draw()

        x1 = int(round(self.lastpix[0], 0))
        y1 = int(round(self.lastpix[1], 0))
        x2 = int(round(event.xdata, 0))
        y2 = int(round(event.ydata, 0))
        # print("( y1 , x1 )=(", y1, ",", x1, ")\n( y2 , x2 )=(", y2, ",", x2, ")")
        def rois2orix(x):
            return x+mincol
        def rois2oriy(y):
            return y+minrow

        img[rois2oriy(y1)][rois2orix(x1)] = img[rois2oriy(y2)][rois2orix(x2)] = [255, 0, 0]

        xgo = int((x2 - x1) / abs(x2 - x1)) if x2 != x1 else None
        ygo = int((y2 - y1) / abs(y2 - y1)) if y2 != y1 else None
        xdis, ydis = abs(x2 - x1), abs(y2 - y1)
        if xdis > ydis:
            while y1 != y2:
                x1 += xgo
                y1 += ygo
                img[rois2oriy(y1)][rois2orix(x1)] = [255, 0, 0]
            while x1 != x2:
                x1 += xgo
                img[rois2oriy(y1)][rois2orix(x1)] = [255, 0, 0]
        else:
            while x1 != x2:
                x1 += xgo
                y1 += ygo
                img[rois2oriy(y1)][rois2orix(x1)] = [255, 0, 0]
            while y1 != y2:
                y1 += ygo
                img[rois2oriy(y1)][rois2orix(x1)] = [255, 0, 0]
        self.lastpix = [event.xdata, event.ydata]


def lineit(p_img, p_img_path, p_rois):
    global img
    global rois
    global minrow,mincol,maxrow,maxcol

    img = p_img
    rois=p_rois
    minrow = rois[0]-10
    mincol = rois[1]-10
    maxrow = rois[2]+10
    maxcol = rois[3]+10

    while minrow<0:
        minrow+=1
    while mincol<0:
        mincol+=1
    while maxrow>=img.shape[0]:
        maxrow-=1
    while maxcol>=img.shape[1]:
        maxcol-=1

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    ax.imshow(copy.copy(img)[minrow:maxrow,mincol:maxcol:])
    handwriting = Handwriting(ax, p_img_path)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.title("按住鼠标左键画线 按鼠标右键结束退出 按滑轮舍弃",fontsize=12)
    plt.show()
    return img

if __name__ == '__main__':
    # 先都画上线
    del_bg_dir=runthis.get_del_bg_dir()
    print(del_bg_dir)
    line_dir=runthis.get_line_dir()
    rois_dir = runthis.get_rois_dir()
    img_paths = next(os.walk(del_bg_dir))[2]
    for img_path in img_paths:
        img = cv2.imread(del_bg_dir + img_path)     # 反常
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 正常
        rois = np.loadtxt(rois_dir + img_path.split('.')[0][:-6] + '.rois', dtype=np.int, delimiter=',').tolist()
        img_line = lineit(img, img_path, rois)  # 画上线的背景是黑色的图

        img_line = cv2.cvtColor(img_line, cv2.COLOR_RGB2BGR)  # 反常
        writename = line_dir + img_path.split('.')[0][:-6] + "_line.png"
        cv2.imwrite(writename, img_line)  # 写入之后正常
        # print(writename,"画线完毕")

    print("图片line处理完毕")
    with open('bad_imgs.txt','r') as f:
        for line in f:
            try:
                os.remove('image_line\\' + line.split('.')[0][:-6] + '_line.png')
            except Exception as e:
                print(line,e)
    os.remove('bad_imgs.txt')
