import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import cv2

def try1():
    p0 = cv2.imread('0.png')
    p1 = cv2.imread('1.png')
    p2 = cv2.imread('2.png')

    p0 = cv2.cvtColor(p0, cv2.COLOR_BGR2RGB)
    p1 = cv2.cvtColor(p1, cv2.COLOR_BGR2RGB)
    p2 = cv2.cvtColor(p2, cv2.COLOR_BGR2RGB)

    plt.subplot(131)
    plt.imshow(p0)
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()

    plt.subplot(132)
    plt.imshow(p1)
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()

    plt.subplot(133)
    plt.imshow(p2)
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0, 0)
    plt.show()

def try2():
    with open('bad_imgs.txt', 'r') as f:
        for line in f:
            try:
                os.remove('image_line\\' + line.split('.')[0][:-6] + '_line.png')
            except Exception as e:
                print(line, e)
    os.remove('bad_imgs.txt')

if __name__ == '__main__':
    try2()