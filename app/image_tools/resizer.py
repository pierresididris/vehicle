#!/usr/bin/python
from PIL import Image
import os, sys

# path = "../dataset/test/Avion/"
# path = "../../reconition/"
# dirs = os.listdir( path )

# def resize():
#     for item in dirs:
#         if os.path.isfile(path+item):
#             im = Image.open(path+item)
#             f, e = os.path.splitext(path+item)
#             imResize = im.resize((64,64), Image.ANTIALIAS).convert('RGB')
#             imResize.save(f + '_resized.jpg', 'JPEG', quality=90)
#
# resize()


def image_resize(image_path):
    if os.path.isfile(image_path):
        print(image_path)
        # im = Image.open(image_path)
        # imResize = im.resize((64, 64), Image.ANTIALIAS).convert('RGB')
        # imResize.save(f + '_resized.jpg', 'JPEG', quality=90)

