# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 03:23:24 2019

@author: Fachry Firdaus
"""

import cv2
import os
import sys
from glob import glob
import numpy as np

def bulk_convert(src, dst, cascade_file = "lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    # Create classifier
    cascade = cv2.CascadeClassifier(cascade_file)
    files = [y for x in os.walk(src) for y in glob(os.path.join(x[0], '*.*'))]
    for image_file in files:
        target_path = "/".join(image_file.strip("/").split('/')[1:-1])
        target_path = os.path.join(dst, target_path) + "/"
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor = 1.1,
                                         minNeighbors = 5,
                                         minSize = (30, 30),
                                         )
        if np.any(faces):
            
            for (x, y, w, h) in faces:
                crop_img = image[y:y+h, x:x+w]
                filename = os.path.basename(image_file).split('.')[0]
                cv2.imwrite(
                    os.path.join(target_path, filename + ".jpg"),
                    crop_img
                )
        else:
            filename = os.path.basename(image_file).split('.')[0]
            cv2.imwrite(
                os.path.join(target_path, filename + ".jpg"),
                image
            )


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: crop.py <source-dir> <target-dir>\n")
        sys.exit(-1)

    bulk_convert(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()