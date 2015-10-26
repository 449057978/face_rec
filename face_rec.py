import numpy as np
import cv2
import lbp
from sklearn import decomposition
import pickle
import sys
from sklearn import svm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

IMAGE_PATH = '../datasets/yalefaces_png/'
TRAIN_LIST = 'subjects_train.lst'
TEST_LIST = 'subjects_test.lst'
COLORS = np.array(['!',
                   '#FF3333',  # red
                   '#0198E1',  # blue
                   '#BF5FFF',  # purple
                   '#FCD116',  # yellow
                   '#FF7216',  # orange
                   '#4DBD33',  # green
                   '#87421F',   # brown
                   '#87421F',   # brown
                   '#87421F',   # brown
                   '#87421F',   # brown
                   '#87421F',   # brown
                   '#87421F'   # brown
                   ])


class FaceRec:

    __block_size = (5, 5)

    def __init__(self):
        # Classifier
        C = 1.0
        self.clf = svm.SVC(kernel='linear', C=C)

        # PCA
        self.pca = decomposition.IncrementalPCA(n_components=3, batch_size=5)

        # Features
        self.features = None
        self.targets = None

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

        X = None
        # X = np.empty((len(img_list),))
        # Read each image for PCA
        for row, img_name in zip(range(len(img_list)), img_list):
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            cv2.imshow('image', img)

            # Calculate the lbp histograms
            features = lbp.circular_lbp(img, 2, 8)
            cv2.imshow('LBP', np.int8(features))

            # separate the image into grids and get histograms
            sizex, sizey = features.shape

            # Set the size
            if row is 0:
                blocksx = sizex/FaceRec.__block_size[0] + 1
                blocksy = sizey/FaceRec.__block_size[1] + 1
                X = np.empty((len(img_list), blocksx * blocksy * 256))

            hists = []
            for i in range(0, sizex, FaceRec.__block_size[0]):
                for j in range(0, sizey, FaceRec.__block_size[1]):
                    hists.append(np.bincount(
                        features[i:i+FaceRec.__block_size[0],
                                 j:j+FaceRec.__block_size[1]].ravel(),
                        minlength=256))
            X[row, :] = (np.array(hists).ravel())

            print X.shape

        # Train the Classifier
        self.pca.fit(X)

        # Storing the data
        self.features = np.empty((len(X), 3))
        for i in range(len(X)):
            self.features[i, :] = self.pca.transform(X[i, :])
        self.targets = y
        print len(self.features)

    def train(self, X=None, y=None):
        if X is None:
            X = self.features
        if y is None:
            y = np.asarray(self.targets)
        print 'train', X.shape
        print 'train y', y
        self.clf.fit(X, y)

    def test(self, img):
        features = lbp.circular_lbp(img, 2, 8)
        hists = []
        sizex, sizey = features.shape
        X = []
        for i in range(0, sizex, FaceRec.__block_size[0]):
                for j in range(0, sizey, FaceRec.__block_size[1]):
                    hists.append(np.bincount(
                        features[i:i+FaceRec.__block_size[0],
                                 j:j+FaceRec.__block_size[1]].ravel(),
                        minlength=256))
        X.append(np.array(hists).ravel())
        X = self.pca.transform(np.array(X))
        prediction = self.clf.predict(X.ravel())
        return prediction

    def pickle(self):
        with open('face_rec.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def unpickle():
        with open('face_rec.pkl', 'rb') as input:
            return pickle.load(input)


def plot_pca(X, target_names):
    fig = plt.figure()
    target_names = np.array(target_names)
    target_names = target_names[:50]
    X = np.array(X)
    X = X[:50, :]
    cats = set(target_names)
    print cats
    ax = Axes3D(fig, elev=-150, azim=110)
    for c, target_name in zip(range(0, len(cats)), set(cats)):
        y = target_names == target_name
        ax.scatter(X[y, 0], X[y, 1],
                   c=COLORS[c+1], cmap=plt.cm.Paired)
    ax.set_title('PCA of faces')


def plot_svm(X, y, clf):
    # create a mesh to plot in
    X = np.array(X)
    if X.shape[1] > 2:
        print "Don't plot"
        return
    y = np.array(y)
    print y
    plt.figure()
    h = 20
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    print x_min, x_max
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = np.empty(Z.shape)
    Y = np.empty(y.shape)
    print set(y)
    i = 0
    for name in set(y):
        z[Z == name] = i
        Y[y == name] = i
        i += 1
    # Put the result into a color plot
    z = z.reshape(xx.shape)
    print Y
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

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
plot_pca(facerec.features, facerec.targets)
plot_svm(facerec.features, facerec.targets, facerec.clf)

# Load a test image
targets = []
test_list = []
with open(IMAGE_PATH + TEST_LIST, 'r') as f:
    while True:
        filename = f.readline()
        # Zero length indicates EOF
        if len(filename) == 0:
            break
        filename = filename.rstrip()
        target = filename.split('.')[0]
        targets.append(target)

        # get filename without return
        test_list.append(IMAGE_PATH + 'test/' + filename)

# Read images for testing
predictions = []
for img_name in test_list:
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Test image', img)
    predictions.append(facerec.test(img))

# get Accuracy
predictions = np.array(predictions)
targets = np.array(targets)
correct = np.count_nonzero(predictions.ravel() == targets)
accuracy = np.float(correct)/len(predictions)
print 'Accuracy: {:.2f}'.format(accuracy)
plt.show()
