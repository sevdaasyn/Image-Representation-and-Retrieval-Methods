import os
import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import normalize



query_path = "dataset/query"
train_path = "dataset/train"

query_file_names = []
for (dirpath, dirnames, filenames) in os.walk(query_path):
    for f in filenames:
        query_file_names.append(os.path.join(os.path.join(dirpath, f)))

train_file_names = []
for (dirpath, dirnames, filenames) in os.walk(train_path):
    for f in filenames:
        train_file_names.append(os.path.join(os.path.join(dirpath, f)))

cv_query_images = []
for filename in query_file_names:
    cv_query_images.append(cv2.imread(filename))

cv_train_images = []
for filename in train_file_names:
    cv_train_images.append(cv2.imread(filename))


def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi/8, np.pi / 320):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros(shape=(1, 40))
    i = 0
    for kern in filters:
        # fimg = ndimage.convolve(img, kern, mode='constant', cval=1.0)
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        mean_val = np.mean(fimg)
        np.put(accum, i, mean_val)
        i = i + 1
    return normalize(accum)

def find_five_most_similar(img):
    response = process(img, filters)
    dist = []
    max_dist_idx = [0, 1, 2, 3, 4]

    for k in range(5):
        dist_in = distance.euclidean(response, train_responces[k])
        dist.append(dist_in)

    dist.sort()
    for k in range(5, len(cv_train_images)):
        dist_in = distance.euclidean(response, train_responces[k])
        for m in range(len(dist)):
            if (dist_in < dist[m]):
                dist[m] = dist_in
                max_dist_idx[m] = k
                break

    print("query file name: ", query_file_names[selected_q])
    for i in range(len(max_dist_idx)):
        print("train {} file name: ".format(str(i)), train_file_names[max_dist_idx[i]])
    print("distances:  ", dist)
    print()
    return max_dist_idx



filters = build_filters()


train_responces = []
for t_img in cv_train_images:
    train_responces.append(process(t_img, filters))


true_output_total = 0
true_output_class = 0
number_of_query = 0


#selected_query_index =20
#find_five_most_similar(cv_query_images[selected_query_index])


for i in range(len(cv_query_images)):

    number_of_query = number_of_query + 1
    response = process(cv_query_images[i], filters)
    dist = 1
    max_dist_idx = 0

    for k in range(len(train_responces)):
        dist_in = distance.euclidean(response, train_responces[k])
        if (dist_in < dist):
            dist = dist_in
            max_dist_idx = k

    if (query_file_names[i].split("_")[0][-3:] == train_file_names[max_dist_idx].split("_")[0][-3:]):
        true_output_total = true_output_total + 1
        true_output_class = true_output_class + 1

    if (i == len(cv_query_images) - 1):
        print("Average class based accuracy for {} =".format(query_file_names[i].split("_")[0][-3:]),
              true_output_class / number_of_query * 100 , "%")
    elif (query_file_names[i + 1].split("_")[0][-3:] != query_file_names[i].split("_")[0][-3:]):
        print("Average class based accuracy for {} =".format(query_file_names[i].split("_")[0][-3:]),
              true_output_class / number_of_query * 100 , "%")
        true_output_class = 0
        number_of_query = 0


print("Average accuracy =", true_output_total / len(cv_query_images) * 100 , "%")







