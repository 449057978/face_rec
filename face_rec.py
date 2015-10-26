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

    def __init__(self):
        # Classifier
        self.clf = svm.LinearSVC()  # one versus rest train only one class

        # PCA
        self.pca = decomposition.PCA(n_components=.5)

        # Features
        self.features = []
        self.targets = []

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
        for img_name in img_list[:20]:
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            cv2.imshow('image', img)

            # Calculate the lbp histograms
            features = lbp.circular_lbp(img, 2, 8)
            cv2.imshow('LBP', np.int8(features))

            # separate the image into grids and get histograms
            block_size = (3, 3)
            sizex, sizey = features.shape

            hists = []
            for i in range(0, sizex, block_size[0]):
                for j in range(0, sizey, block_size[1]):
                    hists.append(np.bincount(
                        features[i:i+block_size[0], j:j+block_size[1]].ravel(),
                        minlength=256))
            X.append(np.array(hists).ravel())
            print np.array(X).shape

        # Train the Classifier
        self.features = self.pca.fit_transform(np.array(X)).tolist()
        self.targets = y[:20]
        print len(self.features)

    def train(self, X=None, y=None):
        if X is None:
            X = np.asarray(self.features)
        if y is None:
            y = np.asarray(self.targets)
        print 'train', X.shape
        print 'train y', y.shape
        self.clf.fit(X, y)

    def test(self, img):
        features = lbp.circular_lbp(img, 2, 8)
        hists = []
        block_size = (3, 3)
        sizex, sizey = features.shape
        X = []
        for i in range(0, sizex, block_size[0]):
                for j in range(0, sizey, block_size[1]):
                    hists.append(np.bincount(
                        features[i:i+block_size[0], j:j+block_size[1]].ravel(),
                        minlength=256))
        X.append(np.array(hists).ravel())
        X = self.pca.transform(np.array(X))
        print X
        prediction = self.clf.predict(X.ravel())
        return prediction

    def pickle(self):
        with open('face_rec.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def unpickle():
        with open('face_rec.pkl', 'rb') as input:
            return pickle.load(input)


# ==================================================
#
# Main
#
# ==================================================
facerec = FaceRec()
loading = False

if len(sys.argv) < 2:
    print '''
        USAGE:
        -t Train the face recognition object
        -l load a face recognition object
        '''
    sys.exit()

if(str(sys.argv[1]) == '-l'):
    loading = True

# PCA
if loading:
    print 'loaded pre-trained human detector'
    facerec = FaceRec.unpickle()
else:
    facerec.get_pca_features()
    facerec.train()
    facerec.pickle()

print facerec.pca

# Load a test image
target = ''
test_list = []
with open(IMAGE_PATH + TEST_LIST, 'r') as f:
                while True:
                    filename = f.readline()
                    # Zero length indicates EOF
                    if len(filename) == 0:
                        break
                    filename = filename.rstrip()
                    target = filename.split('.')[0]

                    # get filename without return
                    test_list.append(IMAGE_PATH + 'test/' + filename)

# Read images for testing
for img_name in test_list[:5]:
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Test image', img)
    prediction = facerec.test(img)
    print prediction
cv2.waitKey(0)




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
