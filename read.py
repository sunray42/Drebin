"""
Reading function for each of the classification problems
read - malware vs non malware
"""

import numpy as np
import sys
import feature_extraction
from os import listdir
from os.path import isfile, join


def read(LOAD_DATA=False):
    if LOAD_DATA:
        print("Previous data not loaded. Attempt to read data ...")
        mypath = "data/feature_vectors"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        print('file list length:', len(onlyfiles))

        print("Reading csv file for ground truth ...")
        ground_truth = np.loadtxt("data/sha256_family.csv", delimiter=",", skiprows=1, dtype=str)

        print("Reading positive and negative texts ...")
        pos = []            #malware sets
        neg = []            #non-malware sets
        for virus in onlyfiles:
            if virus in ground_truth[:, 0]:
                pos.append(virus)
            else:
                if len(neg) < 5560:         #keep malware number == non-malware number
                    neg.append(virus)
                # neg.append(virus)

        print("Extracting features ...")
        x = []          #feature vectors for app
        y = []          #1 for malware, 0 for non-malware
        for text_file in pos:
            sys.stdin = open("%s/%s" % (mypath, text_file))
            features = sys.stdin.readlines()
            sample = feature_extraction.count_feature_set(features)
            x.append(sample)
            y.append(1)

        for text_file in neg:
            sys.stdin = open("%s/%s" % (mypath, text_file))
            features = sys.stdin.readlines()
            sample = feature_extraction.count_feature_set(features)
            x.append(sample)
            y.append(0)

        print("Data is read successfully:")
        x = np.array(x)
        y = np.array(y)
        print(x.shape, y.shape)

        print("Saving data under data_numpy directory ...")
        np.save("balanced_dataset/x_all.npy", x)
        np.save("balanced_dataset/y_all.npy", y)

        return x, y
    else:
        print("Loading previous data ...")
        x_ = np.load("balanced_dataset/x_all.npy")
        y_ = np.load("balanced_dataset/y_all.npy")
        print(x_.shape, y_.shape)
        print(x_[0])
        print(y_[0])
        # print x == x_, y == y_
        return x_, y_

if __name__ == "__main__":
    read(LOAD_DATA=True)