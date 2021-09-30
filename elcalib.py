import numpy as np
from skimage import data
from skimage.io import imshow, imread
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import numpy as np
import scipy

from sklearn.neighbors import NearestNeighbors

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray
import skimage as sk
from glob import glob

from collections import namedtuple

import cv2


MAX_ASPECT_RATIO = 20

def ellipse2conic(r):
    a = r.minor_axis_length
    b = r.major_axis_length
    y0, x0 = r.centroid
    phi = -r.orientation
    R = np.array([[math.cos(phi),-math.sin(phi)],[math.sin(phi),math.cos(phi)]])
    M = R @ np.diag([1/a**2, 1/b**2]) @ R.T
    m = np.array([x0,y0])
    C = np.block([[M, (-M@m).reshape(2,1)],[(-m@M).reshape(1,2), m@M@m-1]])
    return C


def ellipse2ellipsedist(e1,e2):
    # R = 0.092
    R = 1.0
    M1 = ellipse2conic(e1)
    M2 = ellipse2conic(e2)
    C = np.linalg.inv(M1) @ M2
    a3 = np.linalg.det(C)
    a = math.copysign(abs(a3) ** (1 / 3), a3)
    d2 = (3 - np.trace(C)) * R**2/ a
    # assert d2>-1e-15
    # assert d2>0
    d = math.sqrt(abs(d2))
    return d

EllipseStruct = namedtuple("EllipseStruct", "major_axis_length minor_axis_length centroid orientation")

def detect_ellipses(image):
    # MIN_ELLIP_SIZE = image.size // (80 * 80)
    MIN_ELLIP_SIZE = image.size // (150 * 150)
    # MIN_ELLIP_SIZE = image.size // (500 * 500)
    MAX_ELLIP_SIZE = image.size // (8 * 8)
    # apply threshold
    thresh = threshold_otsu(image)

    bw = image < thresh


    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    # plt.imshow(label_image)
    # plt.show()

    ellipse_regions = []
    for r in regionprops(label_image):
        a = r.major_axis_length
        b = r.minor_axis_length
        if r.area>MIN_ELLIP_SIZE and r.area<MAX_ELLIP_SIZE and abs(math.log(a/b)) < math.log(MAX_ASPECT_RATIO):
            ellipse_regions.append(r)
    return ellipse_regions

def distsignatures(D):
    N = len(D)
    signatures = None
    for i in range(N):
        n1, n2, n3 = D[i, :].argsort()[1:4]
        d1,d2,d3 = D[i,[n1,n2,n3]]
        d12 = D[n1, n2]
        d23 = D[n2, n3]
        d13 = D[n3, n1]

        t12 = (d1**2 + d2**2 - d12**2) / (2 * d1 * d2)
        t13 = (d1**2 + d3**2 - d13**2) / (2 * d1 * d3)
        t23 = (d2**2 + d3**2 - d23**2) / (2 * d2 * d3)

        sig = np.array([D[i, n1], D[i, n2], D[i, n3], D[n1, n2], D[n2, n3], D[n3, n1]])
        # sig = np.array([D[i, n1], D[i, n2], D[i, n3]])
        # sig = np.array([D[n1, n2], D[n2, n3], D[n3, n1]])
        # sig = np.array([t12,t13,t23])
        sig = sig / np.linalg.norm(sig)

        if signatures is None:
            signatures = np.zeros((N, len(sig)))
        signatures[i, :] = sig

    return signatures

def detect_pattern(fname, show_image=False):
    # fname="data/house00.png"
    # show_image=True
    print(f"Detecting pattern in {fname}")
    image = rgb2gray(imread(fname))
    p = np.loadtxt('pattern.txt')
    p = p[:, 0:2]
    ellipses = detect_ellipses(image)
    el_centres = np.array([[e.centroid[1], e.centroid[0]] for e in ellipses])
    Ne = len(ellipses)
    Np = len(p)

    Dp = scipy.spatial.distance_matrix(p, p)
    De = np.zeros((Ne, Ne))

    for i, e1 in enumerate(ellipses):
        for j, e2 in enumerate(ellipses):
            De[i, j] = ellipse2ellipsedist(e1, e2)

    signatures_p = distsignatures(Dp)
    signatures_e = distsignatures(De)

    Nmatches = 100
    C = signatures_p @ signatures_e.T

    matches_idx = C.ravel().argsort()[-Nmatches:][::-1]
    idx_p, idx_e = np.unravel_index(matches_idx, C.shape)

    src = p[idx_p]
    tar = el_centres[idx_e]


    homography = sk.transform.ProjectiveTransform
    model, inliers = sk.measure.ransac(data=(src, tar), model_class=homography, min_samples=4, residual_threshold=15,
                                       max_trials=1000)
    if inliers is None or sum(inliers)==0:
        return np.zeros((0,3)), np.zeros((0,2))
    print(f"Ransac#1 inliers: {sum(inliers)} out of {len(inliers)}")

    D = scipy.spatial.distance_matrix(model(p), el_centres)

    b=D.min(axis=1)<5.0
    src = p[b]
    tar = el_centres[D.argmin(axis=1)[b]]
    model, inliers = sk.measure.ransac(data=(src, tar), model_class=homography, min_samples=4, residual_threshold=5,
                                       max_trials=100)
    if inliers is None or sum(inliers)==0:
        return np.zeros((0,3)), np.zeros((0,2))
    print(f"Ransac#2 inliers: {sum(inliers)} out of {len(inliers)}")


    pts = src[inliers]
    pts3d = np.concatenate((pts,np.zeros((len(pts),1))),axis=1)
    pts2d = tar[inliers]


    p_mapped = model(pts)

    if show_image:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=plt.cm.gray)
        for el in ellipses:
            a = el.major_axis_length
            b = el.minor_axis_length
            y0, x0 = el.centroid
            phi = el.orientation
            patch = mpatches.Ellipse((x0, y0), a, b, 90-phi*180/np.pi, fill=False, edgecolor='red', linewidth=0.5)
            ax.add_patch(patch)
        ax.plot(pts2d[:, 0], pts2d[:, 1], 'g+')
        ax.plot(p_mapped[:, 0], p_mapped[:, 1], 'mx')
        plt.show()

    return pts3d, pts2d, C

def detect_pattern_v2(fname, show_image=False):
# fname='data/RDMcalibration/left_57.jpg'
# show_image=True
    print(f"Detecting pattern in {fname}")
    image = rgb2gray(imread(fname))
    p = np.loadtxt('pattern.txt')
    p = p[:, 0:2]
    ellipses = detect_ellipses(image)
    el_centres = np.array([[e.centroid[1], e.centroid[0]] for e in ellipses])
    Ne = len(ellipses)
    Np = len(p)

    Dp = scipy.spatial.distance_matrix(p, p)
    De = np.zeros((Ne, Ne))

    for i, e1 in enumerate(ellipses):
        for j, e2 in enumerate(ellipses):
            De[i, j] = ellipse2ellipsedist(e1, e2)

    quad_e = De.argsort(axis=1)[:, 0:4]
    quad_p = Dp.argsort(axis=1)[:, 0:4]
    nn = NearestNeighbors().fit(el_centres)
    best_inliers=[]
    best_coverage=0
    for q_p in quad_p:
        for q_e in quad_e:
            H = sk.transform.estimate_transform("projective",p[q_p], el_centres[q_e])
            ndist,nind = map(lambda x:x.squeeze(), nn.kneighbors(H(p),1))
            inliers = ndist<10.0
            coverage = len(set(nind[inliers]))
            if coverage>best_coverage:
                best_coverage = coverage
                print(best_coverage)
                best_inliers = inliers
                model = H
    print(f"Brute force: {sum(best_inliers)}")
    ndist,nind = map(lambda x:x.squeeze(), nn.kneighbors(model(p),1))
    best_inliers = ndist<10.0
    for i in range(2):
        model = sk.transform.estimate_transform("projective", p[best_inliers], el_centres[nind[best_inliers]])
        ndist,nind = map(lambda x:x.squeeze(), nn.kneighbors(model(p),1))
        best_inliers = ndist<10.0
        print(f"refinement #{i+1}: {sum(best_inliers)}")

    pts = p[best_inliers]
    pts3d = np.concatenate((pts,np.zeros((len(pts),1))),axis=1)
    pts2d = el_centres[nind[best_inliers]]

    p_mapped = model(pts)

    if show_image:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=plt.cm.gray)
        for el in ellipses:
            a = el.major_axis_length
            b = el.minor_axis_length
            y0, x0 = el.centroid
            phi = el.orientation
            patch = mpatches.Ellipse((x0, y0), a, b, 90-phi*180/np.pi, fill=False, edgecolor='red', linewidth=0.5)
            ax.add_patch(patch)
        ax.plot(pts2d[:, 0], pts2d[:, 1], 'g+')
        ax.plot(p_mapped[:, 0], p_mapped[:, 1], 'mx')
        plt.show()
    return pts3d, pts2d

def calibrate_sequence(seq_path):
    fnames = sorted(glob(seq_path))
    all_pts3d=[]
    all_pts2d=[]
    for fname in fnames:
        pts3d, pts2d = detect_pattern_v2(fname,show_image=True)
        if len(pts3d)>0:
            all_pts3d.append(pts3d.astype(np.float32))
            all_pts2d.append(pts2d.astype(np.float32))
    image = rgb2gray(imread(fnames[0]))
    h,  w = image.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(all_pts3d, all_pts2d, (w,h), None, None)

    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    return mtx, dist

if __name__ == '__main__':
    K1, D1 = calibrate_sequence("data/RDMcalibration/left_*.jpg")
    K2, D2 = calibrate_sequence("data/RDMcalibration/right_*.jpg")

    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objp, leftp, rightp, K1, D1, K2, D2, image_size)