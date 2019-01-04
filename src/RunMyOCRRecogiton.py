import pickle
import train
import test
from sklearn.metrics import confusion_matrix
from skimage import io, exposure
import matplotlib.pyplot as plt


def main(path, fileName):

    pkl_file = open('test1_gt.pkl', 'rb')
    mydict = pickle.load(pkl_file)
    pkl_file.close()
    classes = mydict['classes']
    locations = mydict['locations']

    # showImage,showHistogram,showBinaryImage,showImageLabel, showBoundingBoxes,showDistanceMatrix
    train.main([True, True, True, True, True, True])
    test.main(classes, locations, train.Features, path, fileName)

# replace this with your path and filename
main('test1.bmp', 'test1')