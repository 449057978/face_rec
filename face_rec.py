import numpy as np
import cv2
import lbp
from sklearn import decomposition
import pickle
import sys
from sklearn import svm
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

IMAGE_PATH = '../datasets/yalefaces_png/'
TRAIN_LIST = 'subjects_train.lst'
TEST_LIST = 'subjects_test.lst'


class FaceRec:

    # Classifier
    clf = svm.LinearSVC()  # one versus rest train only one class

    # PCA
    pca = decomposition.PCA(n_components=3)

    # Features
    features = 0
    targets = 0

    def get_pca_features(self):
        # Get the list of images
        img_list = []
        y = []
        with open(IMAGE_PATH + TRAIN_LIST, 'r') as f:
                while True:
                    filename = f.readline()
                    # Zero length indicates EOF
                    if len(filename) == 0:
                        break
                    filename = filename.rstrip()
                    target = filename.split('.')
                    y.append(target[0])

                    # get filename without return
                    img_list.append(IMAGE_PATH + 'train/' + filename)

        X = []
        # Read each image for PCA
        for img_name in img_list[:10]:
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            cv2.imshow('image', img)

            # Calculate the lbp histograms
            features = lbp.circular_lbp(img)
            cv2.imshow('LBP', np.int8(features))

            # separate the image into grids and get histograms
            block_size = (3, 3)
            sizex, sizey = features.shape

            for i in range(0, sizex, block_size[0]):
                for j in range(0, sizey, block_size[1]):
                    X.append(np.bincount(
                        features[i:i+block_size[0], j:j+block_size[1]].ravel(),
                        minlength=256))

        print np.array(X).shape
        self.features = np.array(X)
        print self.X
        # Train the Classifier
        self.pca.fit(np.array(X))

    def train(self, X=features, y=targets):
        X = np.asarray(X)
        print X
        y = np.asarray(y)
        self.clf.fit(X, y)

    def test(self, img):
        features = lbp.circular_lbp(img)
        X = self.pca.transfrom(features)
        prediction = self.clf.predict(X.ravel())
        return prediction


# ==================================================
#
# Main
#
# ==================================================
facerec = FaceRec()
loading = False

if(str(sys.argv[1]) == '-l'):
    loading = True

# PCA
if loading:
    print 'loaded pre-trained human detector'
    with open('face_rec_pca.pkl', 'rb') as input:
        facerec.pca = pickle.load(input)
    with open('face_rec_X.pkl', 'rb') as input:
        facerec.features = pickle.load(input)
else:
    facerec.get_pca_features()
    with open('face_rec_pca.pkl', 'wb') as output:
            pickle.dump(facerec.pca, output, pickle.HIGHEST_PROTOCOL)
    with open('face_rec_X.pkl', 'wb') as output:
        pickle.dump(facerec.features, output, pickle.HIGHEST_PROTOCOL)

# Train the classifier
facerec.train()

# Load a test image

print facerec.pca

# Read images for testing=



# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

# ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.spectral)

# x_surf = [X[:, 0].min(), X[:, 0].max(),
#           X[:, 0].min(), X[:, 0].max()]
# y_surf = [X[:, 0].max(), X[:, 0].max(),
#           X[:, 0].min(), X[:, 0].min()]
# x_surf = np.array(x_surf)
# y_surf = np.array(y_surf)
# v0 = facerec.pca.transform(facerec.pca.components_[0])
# v0 /= v0[-1]
# v1 = facerec.pca.transform(facerec.pca.components_[1])
# v1 /= v1[-1]

# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])

# plt.show()
