#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: Your Name
# Created Time : Mon 24 Jul 2023 03:20:36 PM CST
# File Name: face_detect.py
# Description:
"""
import cv2
import os

class FaceDetectCV2(object):
    def __init__(self, model_rootpath, cascade_file=''):
        model_path = os.path.join(model_rootpath, cascade_file)
        self.detector = cv2.CascadeClassifier(model_path)

    def detect_faces(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.detector.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
        res = []
        for face in faces:
            print(face)
            temp = [face[0], face[1], face[0]+face[2], face[1]+face[3]]
            print(temp)
            res.append(temp)
        return res

def test():
    cascade_file = 'lbpcascade_animeface.xml'
    model_rootpath = '/algorithm/zhaoweisong/SadTalker/checkpoints'
    detector = FaceDetectCV2(model_rootpath, cascade_file)
    import sys
    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)
    print(faces)

if __name__ == '__main__':
    test()
