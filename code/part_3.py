import os
import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans

query_path  = "dataset/query"
train_path  = "dataset/train"


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
    cv_query_images.append(cv2.imread( filename))


cv_train_images = []
for filename in train_file_names:
    cv_train_images.append(cv2.imread( filename))



def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_img, None)
    return (kp, desc)


def make_average_descriptor(img_desc):
    return np.average(img_desc, axis=0)

def make_desc_arr(cv_train_images):
    desc_array = []
    for k in range(5):
        train_gray = to_gray(cv_train_images[k])
        for d in range(128):
            desc_array.append(gen_sift_features(train_gray)[1][d])
    return desc_array


def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros( len(cluster_alg.cluster_centers_) )
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram




descriptor_list = make_desc_arr(cv_train_images)
kmeans = KMeans(n_clusters = 500)
kmeans.fit(descriptor_list)

preprocessed_trains =[]
for i in range(len(cv_train_images)):
    train_gray = to_gray(cv_train_images[i])
    train_kp, train_desc = gen_sift_features(train_gray)
    histogram = build_histogram(train_desc, kmeans)
    preprocessed_trains.append(histogram)

def find_five_most_similar(img, selected_q ):
    query_gray = to_gray(img)
    query_kp, query_desc = gen_sift_features(query_gray)

    histogram = build_histogram(query_desc, kmeans)
    dist = []
    max_dist_idx = [0, 1, 2, 3, 4]

    for k in range(5):
        dist_in = distance.euclidean(histogram, preprocessed_trains[k])
        dist.append(dist_in)

    dist.sort()
    for k in range(5, len(cv_train_images)):
        dist_in = distance.euclidean(histogram, preprocessed_trains[k])
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

# selected_q = 17
# find_five_most_similar(cv_query_images[selected_q] ,selected_q)


true_output_total =0
true_output_class =0
number_of_query=0
for i in range(len(cv_query_images)):
    number_of_query = number_of_query + 1

    query_gray = to_gray(cv_query_images[i])
    query_kp , query_desc = gen_sift_features(query_gray)

    histogram = build_histogram(query_desc, kmeans)
    dist = distance.euclidean(histogram, preprocessed_trains[0])
    max_dist_idx = 0

    for k in range(1,len(cv_train_images)):
        dist_in = distance.euclidean(histogram, preprocessed_trains[k])
        if (dist_in < dist):
            dist = dist_in
            max_dist_idx = k

    if (query_file_names[i].split("_")[0][-3:] == train_file_names[max_dist_idx].split("_")[0][-3:]):
        true_output_total = true_output_total + 1
        true_output_class = true_output_class + 1

    if (i == len(cv_query_images) - 1):
        print("Average class based accuracy: {} = %".format(query_file_names[i].split("_")[0][-3:]),
              true_output_class / number_of_query * 100)
    elif (query_file_names[i + 1].split("_")[0][-3:] != query_file_names[i].split("_")[0][-3:]):
        print("Average class based accuracy: {} = %".format(query_file_names[i].split("_")[0][-3:]),
              true_output_class / number_of_query * 100)
        true_output_class = 0
        number_of_query = 0

print("Average accuracy: %", true_output_total / len(cv_query_images) * 100)



cv2.waitKey(0)
cv2.destroyAllWindows()
