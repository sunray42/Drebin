"""
Reading function for each of the classification problems
read - malware vs non malware
"""

import numpy as np
import sys
from os import listdir
from os.path import isfile, join

FEATURES_SET = {
    "feature": 1,
    "permission": 2,
    "activity": 3,
    "service_receiver": 3,
    "provider": 3,
    "service": 3,
    "intent": 4,
    "api_call": 5,
    "real_permission": 6,
    "call": 7,
    "url": 8
}

"""
Create feature vectors
"""
def count_feature_set(lines):
    """
    Count how many features belong to a specific set
    :param lines: features in the text file
    :return:
    """
    features_map = {x: 0 for x in range(1, 9)}
    for l in lines:
        if l != "\n":
            set = l.split("::")[0]
            features_map[FEATURES_SET[set]] += 1
    features = []
    for i in range(1, 9):
        features.append(features_map[i])
    return features

def undersampling(LOAD_DATA=False, neg_pos_ratio = 1):
    if LOAD_DATA:
        print("Previous data not loaded. Attempt to read data ...")
        mypath = "../data/feature_vectors"
        softwares = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        print('file list length:', len(softwares))

        print("Reading csv file for malwares ...")
        malwares = np.loadtxt("../data/sha256_family.csv", delimiter=",", skiprows=1, dtype=str)
        num_malwares = malwares.shape[0]
        print("num of malwares: ", num_malwares)

        print("Reading positive and negative software info ...")
        pos = []            #malware sets
        neg = []            #non-malware sets
        for software in softwares:
            if software in malwares[:, 0]:
                pos.append(software)
            else:
                if len(neg) < num_malwares * neg_pos_ratio:
                    neg.append(software)

        print("Extracting features ...")
        x = []          #feature vectors for app
        y = []          #1 for malware, 0 for non-malware
        for text_file in pos:
            sys.stdin = open("%s/%s" % (mypath, text_file))
            features = sys.stdin.readlines()
            sample = count_feature_set(features)
            x.append(sample)
            y.append(1)

        for text_file in neg:
            sys.stdin = open("%s/%s" % (mypath, text_file))
            features = sys.stdin.readlines()
            sample = count_feature_set(features)
            x.append(sample)
            y.append(0)

        print("Data is read successfully:")
        x = np.array(x).T
        y = np.array(y).reshape((1, len(y)))
        print("x shape: ", x.shape, " y shape: ", y.shape)

        print("Saving data under data_numpy directory ...")
        np.save("npratio_" + str(neg_pos_ratio) + "_dataset/x_all.npy", x)
        np.save("npratio_" + str(neg_pos_ratio) + "_dataset/y_all.npy", y)

        return x, y
    else:
        print("Loading previous data ...")
        x_ = np.load("npratio_" + str(neg_pos_ratio) + "_dataset/x_all.npy")
        y_ = np.load("npratio_" + str(neg_pos_ratio) + "_dataset/y_all.npy")
        print("x shape: ", x_.shape, " y shape: ", y_.shape)
        print("x[0]: ", x_[0])
        print("y[0]: ", y_[0])
        return x_, y_

def oversampling(LOAD_DATA=False, repeat_sampling_times = 1):
    if LOAD_DATA:
        print("Previous data not loaded. Attempt to read data ...")
        mypath = "../data/feature_vectors"
        softwares = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        print('file list length:', len(softwares))

        print("Reading csv file for malwares ...")
        malwares = np.loadtxt("../data/sha256_family.csv", delimiter=",", skiprows=1, dtype=str)
        num_malwares = malwares.shape[0]
        print("num of malwares: ", num_malwares)

        print("Reading positive and negative software info ...")
        pos = []            #malware sets
        neg = []            #non-malware sets
        for software in softwares:
            if software in malwares[:, 0]:
                pos.append(software)
            else:
                neg.append(software)

        print("Extracting features ...")
        x = []          #feature vectors for app
        y = []          #1 for malware, 0 for non-malware
        for text_file in pos:
            sys.stdin = open("%s/%s" % (mypath, text_file))
            features = sys.stdin.readlines()
            sample = count_feature_set(features)
            x.append(sample)
            y.append(1)
        x *= repeat_sampling_times
        y *= repeat_sampling_times

        for text_file in neg:
            sys.stdin = open("%s/%s" % (mypath, text_file))
            features = sys.stdin.readlines()
            sample = count_feature_set(features)
            x.append(sample)
            y.append(0)

        print("Data is read successfully:")
        x = np.array(x).T
        y = np.array(y).reshape((1, len(y)))
        print("x shape: ", x.shape, " y shape: ", y.shape)

        print("Saving data under data_numpy directory ...")
        np.save("resamptimes_" + str(repeat_sampling_times) + "_dataset/x_all.npy", x)
        np.save("resamptimes_" + str(repeat_sampling_times) + "_dataset/y_all.npy", y)

        return x, y
    else:
        print("Loading previous data ...")
        x_ = np.load("resamptimes_" + str(repeat_sampling_times) + "_dataset/x_all.npy")
        y_ = np.load("resamptimes_" + str(repeat_sampling_times) + "_dataset/y_all.npy")
        print("x shape: ", x_.shape, " y shape: ", y_.shape)
        print("x[0]: ", x_[0])
        print("y[0]: ", y_[0])
        return x_, y_

if __name__ == "__main__":
    # undersampling(LOAD_DATA=True, neg_pos_ratio=5)
    oversampling(LOAD_DATA=True, repeat_sampling_times=1)