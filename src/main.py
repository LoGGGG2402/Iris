import os
from math import sqrt
from os import listdir
from os.path import isdir, join

import cv2

from src.segmentation import iris_segmentation


def detect_iris():
    source = 'CASIA1'
    # get all folders in source
    folders = [f for f in listdir(source) if isdir(join(source, f))]

    for folder in folders:
        eyes_img = [f for f in listdir(join(source, folder)) if f.endswith('.jpg')]

        for eye_img in eyes_img:
            img = cv2.imread(join(source, folder, eye_img))
            mask = iris_segmentation(img)
            if mask is not None:
                cv2.imshow('mask', mask)
                cv2.waitKey(0)
            else:
                print("no mask")
                cv2.waitKey(0)


def main():
    # img = cv2.imread('CASIA1/1/001_2_2.jpg')
    # mask = iris_segmentation(img)
    # if mask is not None:
    #     cv2.imshow('mask', mask)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # else:
    #     print("no mask")
    #     cv2.waitKey(0)
    # img = cv2.imread('CASIA1/1/001_2_2.jpg')
    #
    # mask = iris_segmentation(img)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    detect_iris()

if __name__ == '__main__':
    main()
