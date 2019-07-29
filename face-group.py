# Copyright (c) 2019 Hiroki Takemura (kekeho)
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import face_recognition
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import glob
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

filelist = glob.glob('./faces/*[jpg|JPG|png]')
images = [face_recognition.load_image_file(i) for  i in filelist]

faces = []
for i in images:
#     locations = face_recognition.face_locations(i, model='cnn')  # using cnn
    faces += face_recognition.face_encodings(i)

faces = np.array(faces)
compressed = PCA(n_components=3).fit_transform(faces)
# compressed = TSNE(n_components=2, random_state=0, method='exact').fit_transform(n_10)

db = DBSCAN(eps=0.45, min_samples=1).fit(faces)
labels = db.labels_
cluster_max = np.max(labels)

fig = plt.figure()
ax = Axes3D(fig)

#  グループごと点をプロット & print
for i in range(0, cluster_max+1):
    points = compressed[labels == i]
    # plt.scatter(points[:, 0], points[:, 1, points[:, 2]])
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
    print('\nGroup', i)
    for i, v in enumerate(labels == i):
        if v == True:
            print(filelist[i])


# ファイル名をプロット
for i, (x, y, z) in enumerate(compressed):
    ax.text(x, y, z, filelist[i].split('/')[-1])

points = compressed[labels == -1]
ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], color="gray")
print('\nNoise')
for i, v in enumerate(labels == -1):
    if v == True:
        print(filelist[i])

plt.show()
