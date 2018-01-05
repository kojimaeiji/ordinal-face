'''
Created on Jan 5, 2018

@author: saboten
'''
import unittest


import cv2


def read_oneimg(filename):
    img = cv2.imread(filename)
    print(img.shape)


def load_data():
    # depends on datasize
    pass


def preprocess():
    # crop or resize
    pass


class Test(unittest.TestCase):

    def testName(self):
        read_oneimg('../misc/s_scattest.jpeg')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
