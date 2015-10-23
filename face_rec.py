import numpy as np
import cv2
import sklearn

IMAGE_PATH = '../datasets/yalefaces_png/'
LIST_NAME = 'subjects.lst'

clf = sklearn.svm.LinearSVC()
features = 0
y = 0


def circular_lbp(img):
    pass

# Get the list of images
img_list = []
with open(IMAGE_PATH + LIST_NAME, 'r') as f:
        while True:
            filename = f.readline()
            # Zero length indicates EOF
            if len(filename) == 0:
                break
            filename = filename.rstrip()
            # get filename without return
            img_list.append(IMAGE_PATH + filename)

# Read each image for training
for img_name in img_list:
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)
    cv2.waitKey(0)

    # Calculate the lbp histograms
    features = circular_lbp(img)

    # Train the Classifier
clf.fit(features, y)

# Read images for testing
# Test the classifier
