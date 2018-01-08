'''
Created on Jan 5, 2018

@author: saboten
'''
import unittest

import sys
import cv2
import os
from scipy.io import loadmat

def read_oneimg(filename):
    img = cv2.imread(filename)
    print(img.shape)


def load_and_shurink_data(inputfile, outfile,num=100):
    d = loadmat(inputfile)
    d = d['imdb'][0, 0][:num]

    print(len(d))
    savemat(outfile, num)


def preprocess(input_dir, output_dir):
    # crop or resize
    pass


if __name__ == "__main__":
    #input_dir = sys.argv[1]
    #output_dir = sys.argv[2]
    #num = sys.argv[3]
    input_dir = '/home/jiman/facedata/imdb/imdb.mat'
    output_dir = '/home/jiman/facedata/imdb/imdb_out.mat'
    num = 100
    load_and_shurink_data(input_dir,
                          output_dir, 
                          num) 
    
