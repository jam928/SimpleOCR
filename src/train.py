import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.morphology import erosion, dilation, opening, closing, disk, remove_small_objects
import pickle
from scipy import ndimage as ndi
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_mean

Features = []
label_img = []

def file_read(path, fname, show, th_value):
    img = io.imread(path)
    # print img.shape
    if show[0]:
        visualize_matrix(img, fname)
    if show[1]:
        histogram(img,fname)
    apply_threshold(img,fname,show, th_value)

def visualize_matrix(img,fname):
    io.imshow(img)
    plt.title('Original Image')
    io.show()

def histogram(img,fname):
    hist = exposure.histogram(img)
    plt.bar(hist[1], hist[0])
    plt.title('Histogram')
    plt.show()

def apply_threshold(img,fname,show, th_value):
    global Features, label_img
    th = th_value
    img_binary = (img < th).astype(np.double)
    if show[2]:
        io.imshow(img_binary)
        plt.title('Binary Image')
        io.show()
    img_label = label(img_binary, background=0)
    if show[3]:
        io.imshow(img_label)
        plt.title('Labeled Image')
        io.show()
    print fname + str(th)
    print np.amax(img_label)


    regions = regionprops(img_label)
    io.imshow(img_binary)
    ax = plt.gca()
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
        Features.append(hu)
        label_img.append(fname)

    if show[4]:
        ax.set_title('Bounding Boxes')
        plt.savefig('bounding_boxes_image_'+fname + '.png')
        io.show()
    else:
        ax.cla()

def show_distance_confusion_matrix(fname):
    global Features, label_img
    #normalize
    scaler = StandardScaler()
    f2 = scaler.fit_transform(Features)

    #show distance matrix
    D = cdist(f2, f2)
    io.imshow(D)
    plt.title('Distance Matrix of ' + fname)
    plt.savefig('distance_matrix_' + fname)
    io.show()

    # show confusion matrix
    D_index = np.argsort(D, axis=1)
    ypred = []

    for i in range(len(D_index)):
        ypred.append(label_img[D_index[i][1]])

    conFm = confusion_matrix(label_img, ypred)
    io.imshow(conFm)
    plt.title("Confusion Matrix")
    plt.savefig('Confusion Matrix of testing' + '.png')
    io.show()


def main(show):

    threshold_values = [185, 175, 175, 200, 215, 195, 180, 185, 185, 180]
    file_read('a.bmp', 'a', show, threshold_values[0])
    file_read('d.bmp', 'd', show, threshold_values[1])
    file_read('m.bmp', 'm', show, threshold_values[2])
    file_read('n.bmp', 'n', show, threshold_values[3])
    file_read('o.bmp', 'o', show, threshold_values[4])
    file_read('p.bmp', 'p', show, threshold_values[5])
    file_read('q.bmp', 'q', show, threshold_values[6])
    file_read('r.bmp', 'r', show, threshold_values[7])
    file_read('u.bmp', 'u', show, threshold_values[8])
    file_read('w.bmp', 'w', show, threshold_values[9])

    if show[5]:
        show_distance_confusion_matrix('training')
