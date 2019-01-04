import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.morphology import erosion, dilation, opening, closing, disk
from sklearn.preprocessing import StandardScaler
import pickle
import train

# Features = []

def file_read(path, fname,classes, locations, Features):
    img = io.imread(path)
    print img.shape
    visualize_matrix(img)
    histogram(img)
    apply_threshold(img,fname, classes,locations, Features)

def visualize_matrix(img):
    io.imshow(img)
    plt.title('Original Image')
    io.show()

def histogram(img):
    hist = exposure.histogram(img)
    plt.bar(hist[1], hist[0])
    plt.title('Histogram')
    plt.show()

def apply_threshold(img,fname, classes, locations, Features):
    th = 230
    img_binary = (img < th).astype(np.double)
    io.imshow(img_binary)
    plt.title('Binary Image')
    io.show()

    img_label = label(img_binary, background=0)
    io.imshow(img_label)
    plt.title('Labeled Image')
    io.show()
    print np.amax(img_label)

    regions = regionprops(img_label)
    io.imshow(img_binary)
    ax = plt.gca()

    ypred = []

    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        roi = img_binary[minr:maxr, minc:maxc]
        m = moments(roi)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        mu = moments_central(roi, cr, cc)
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        pred_coord = np.array(list(props.centroid))
        ypred.append(np.flipud(pred_coord))

    ax.set_title('Bounding Boxes')
    plt.savefig('bounding_boxes_image_'+fname + '.png')
    io.show()

    t = np.array(ypred)
    x = t.astype(int)
    D = cdist(x, locations)
    print_matrix(D)

def print_matrix(D):
    scaler = StandardScaler()
    f2 = scaler.fit_transform(D)
    D = cdist(f2, f2)
    io.imshow(D)
    plt.title('Distance Matrix')
    plt.savefig('distance_matrix_test1.png')
    io.show()
    # print len(Features)


def main(classes, locations, Features,path, fileName):
    file_read(path, fileName, classes, locations, Features)

    # normalized()
