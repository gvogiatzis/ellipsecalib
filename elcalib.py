import argparse,textwrap

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # top level parser
    subparsers = parser.add_subparsers(help="Calibration commands:", dest='cmd')
    parser_cal = subparsers.add_parser("cal", description='Compute intrinsic parameters of a camera from a sequence of images of a calibration pattern.',
                                       help="Calibrate from image sequence. Will output calibration matrix K, the unistorted K matrix and the radial distortion parameters D.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_und = subparsers.add_parser("und", description='Remove the effect of radial distortion from a set of images',
                                       help="Undistort an image sequence. The output images will have a 'und_' prefix.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_ste = subparsers.add_parser("ste", help="Calibrate a stereo-rig  from image sequences. Will output the two K matrices, the R-t transformation between cameras and the rad distortion coefficients D1 and D2. Can use existing monocular calibrations to initialize.",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                       description='Calibrate a stereo rig using images taken from left and right camera, of a calibration pattern.',
                                       epilog="Notes: (1) You should make sure that the left and right sequences have the same number of images and these images are sorted so that pairs of elements from each list correspond to the same frame.  (2) If the left and right camera files exist, they will be used to constrain the optimisation. Otherwise the filenames are used to output the computed camera matrices.")


    # Single camera calibration
    parser_cal.add_argument("image", type=str,  nargs="+",
                        help="One or more of the calib pattern image filenames. Wildcards should work, e.g 'img*.png'")
    parser_cal.add_argument("-k", "--kmatrix", type=str,default="k.txt",
                        help="The output filename where the  calibration matrix will be written")
    parser_cal.add_argument("-u", "--undkmatrix", type=str,default="newk.txt",
                        help="The output filename where the undistorted calibration matrix will be written")
    parser_cal.add_argument("-d", "--dmatrix", type=str,default="d.txt",
                        help="The output filename where the  distortion coefficients will be written")
    parser_cal.add_argument("-t", "--threshold", type=float, default=5.0,
                        help="The threshold reprojection distance for inlier pattern dots.")
    parser_cal.add_argument("-s", "--showimage", action='store_true',
                        help="Will render each pattern image with the detected dots.")

    # Warp images to remove radial distortion
    parser_und.add_argument("image", type=str,  nargs="+",
                        help="One or more image filenames to undistort given the parameters provided. Wildcards should work, e.g 'img*.png'")
    parser_und.add_argument("-k", "--kmatrix", type=str, default="k.txt",
                            help="Filename of textfile that holds K matrix")
    parser_und.add_argument("-u", "--undkmatrix", type=str,default="newk.txt",
                        help="Filename of textfile that holds undistorted calibration matrix")
    parser_und.add_argument("-d", "--dmatrix", type=str, default="d.txt",
                            help="Filename of textfile that holds radial distortion parameters")

    # stereo camera rig calibration
    parser_ste.add_argument("-l","--left", type=str,  nargs="+",
                        help="One or more of the left pattern image filenames. Wildcards should work, e.g 'img*.png'",
                        required=True)
    parser_ste.add_argument("-r","--right", type=str,  nargs="+",
                        help="One or more of the right pattern image filenames. Wildcards should work, e.g 'img*.png'",
                        required=True)
    parser_ste.add_argument("-t", "--threshold", type=float, default=5.0,
                        help="The threshold reprojection distance for inlier pattern dots.")
    parser_ste.add_argument("-s", "--showimage", action='store_true',
                        help="Will render each pattern image with the detected dots.")
    parser_ste.add_argument("-rk", "--rightkmatrix", type=str,default="rightk.txt",
                        help="Name of textfile that stores the right calibration matrix.")
    parser_ste.add_argument("-lk", "--leftkmatrix", type=str,default="leftk.txt",
                        help="Name of textfile that stores the left calibration matrix")
    parser_ste.add_argument("-ld", "--leftdmatrix", type=str,default="leftd.txt",
                        help="Name of textfile that stores the left lin. distortion params matrix")
    parser_ste.add_argument("-rd", "--rightdmatrix", type=str,default="rightd.txt",
                        help="Name of textfile that stores the right lin. distortion params matrix")

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_usage()
        exit(0)


import numpy as np
from skimage import data
from skimage.io import imshow, imread,imsave
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import numpy as np
import scipy
import os.path

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
        if MIN_ELLIP_SIZE< r.area <MAX_ELLIP_SIZE and b>1 and abs(math.log(a/b)) < math.log(MAX_ASPECT_RATIO):
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

def detect_pattern(fname, show_image=False,inlier_threshold=5.0):
    print(f"******* Detecting pattern in {fname} *******")
    image = rgb2gray(imread(fname))
    p = np.loadtxt('pattern.txt')
    p = p[:, 0:2]
    ellipses = detect_ellipses(image)
    print(f"{len(ellipses)} dots detected in image.")

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
            inliers = ndist<inlier_threshold
            coverage = len(set(nind[inliers]))
            if coverage>best_coverage:
                best_coverage = coverage
                print(f"\rPattern dots found: {best_coverage} out of {Np}",end='')
                best_inliers = inliers
                model = H
    ndist,nind = map(lambda x:x.squeeze(), nn.kneighbors(model(p),1))
    best_inliers = ndist<inlier_threshold
    print("")
    for i in range(2):
        model = sk.transform.estimate_transform("projective", p[best_inliers], el_centres[nind[best_inliers]])
        ndist,nind = map(lambda x:x.squeeze(), nn.kneighbors(model(p),1))
        best_inliers = ndist<inlier_threshold
        print(f"Pattern dots found after refinement #{i+1}: {sum(best_inliers)} out of {Np}")

    print("")
    pts3d = np.concatenate((p,np.zeros((len(p),1))),axis=1)
    pts2d = el_centres[nind]

    p_mapped = model(p)

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
        ax.plot(pts2d[best_inliers, 0], pts2d[best_inliers, 1], 'g+')
        ax.plot(p_mapped[best_inliers, 0], p_mapped[best_inliers, 1], 'mx')
        plt.show()

    return pts3d, pts2d, best_inliers

def calibrate_sequence(fnames,inlier_threshold=5.0, show_image=False):
    print("")
    fnames = sorted(fnames)
    all_pts3d=[]
    all_pts2d=[]
    for fname in fnames:
        pts3d, pts2d, inliers = detect_pattern(fname,show_image=show_image,inlier_threshold=inlier_threshold)
        pts3d = pts3d[inliers]
        pts2d = pts2d[inliers]
        if len(pts3d)>0:
            all_pts3d.append(pts3d.astype(np.float32))
            all_pts2d.append(pts2d.astype(np.float32))
    image = rgb2gray(imread(fnames[0]))
    h,  w = image.shape[:2]
    # print(all_pts3d)
    # print(all_pts2d)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(all_pts3d, all_pts2d, (w,h), None, None)
    print(f"--- Final Reprojection error: {ret} ---\n")

    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    return mtx, newcameramtx, dist

def calibrate_stereo_rig(leftfnames,rightfnames,inlier_threshold=5.0, show_image=False,K1=None, D1=None, K2=None, D2=None):
    all_pts3d=[]
    all_pts2d_L=[]
    all_pts2d_R=[]
    for leftimg, rightimg in zip(leftfnames, rightfnames):
        pts3d_L, pts2d_L, inliers_L = detect_pattern(leftimg, show_image=show_image, inlier_threshold=inlier_threshold)
        pts3d_R, pts2d_R, inliers_R = detect_pattern(rightimg, show_image=show_image, inlier_threshold=inlier_threshold)
        inliers = inliers_L & inliers_R
        print(f"{sum(inliers)} dots visible in both left and right cameras")
        pts3d = pts3d_L[inliers] # could have used pts3d_R[inliers], they are the same.
        pts2d_L = pts2d_L[inliers]
        pts2d_R = pts2d_R[inliers]
        all_pts3d.append(pts3d.astype(np.float32))
        all_pts2d_L.append(pts2d_L.astype(np.float32))
        all_pts2d_R.append(pts2d_R.astype(np.float32))
    image = rgb2gray(imread(leftfnames[0]))
    h, w = image.shape[:2]
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(all_pts3d, all_pts2d_L, all_pts2d_R, cameraMatrix1=K1, distCoeffs1=D1, cameraMatrix2=K2, distCoeffs2=D2, imageSize=(w,h), flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    # ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(all_pts3d, all_pts2d_L, all_pts2d_R, cameraMatrix1=K1, distCoeffs1=D1, cameraMatrix2=K2, distCoeffs2=D2, imageSize=(w,h), flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    # ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(all_pts3d, all_pts2d_L, all_pts2d_R, cameraMatrix1=K1, distCoeffs1=D1, cameraMatrix2=K2, distCoeffs2=D2, imageSize=(w,h), flags=cv2.CALIB_FIX_INTRINSIC)
    # ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(all_pts3d, all_pts2d_L, all_pts2d_R, cameraMatrix1=None, distCoeffs1=None, cameraMatrix2=None, distCoeffs2=None, imageSize=(w,h))
    # ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(all_pts3d, all_pts2d_L, all_pts2d_R, K1, D1, K2, D2, (w,h))
    print(f"--- Final Reprojection error: {ret} ---\n")

if __name__ == '__main__':
    if args.cmd == "cal":
        imfilenames = sorted(args.image)
        show_image = args.showimage
        inlier_threshold = args.threshold
        K, uK, D = calibrate_sequence(imfilenames, inlier_threshold = inlier_threshold, show_image = show_image)
        np.savetxt(args.kmatrix, K)
        np.savetxt(args.undkmatrix, uK)
        np.savetxt(args.dmatrix, D)
    elif args.cmd == "und":
        K = np.loadtxt(args.kmatrix)
        uK = np.loadtxt(args.undkmatrix)
        D = np.loadtxt(args.dmatrix)
        imfilenames = sorted(args.image)
        for fname in imfilenames:
            print(f"Undistorting {fname}.")
            image = imread(fname)
            image_und = cv2.undistort(image, K, D, None, uK)
            imsave("und_" + fname, image_und)
    elif args.cmd == "ste":
        if len(args.left)!=len(args.right):
            print("Left and right sequences must have the same number of images.")
            exit(0)
        leftimgs = sorted(args.left)
        rightimgs = sorted(args.right)
        show_image = args.showimage
        inlier_threshold = args.threshold


        Kright = np.loadtxt(args.rightkmatrix) if os.path.exists(args.rightkmatrix) else None
        Kleft = np.loadtxt(args.leftkmatrix) if os.path.exists(args.leftkmatrix) else None
        Dright = np.loadtxt(args.rightdmatrix) if os.path.exists(args.rightdmatrix) else None
        Dleft = np.loadtxt(args.leftdmatrix) if os.path.exists(args.leftdmatrix) else None
        calibrate_stereo_rig(leftimgs, rightimgs, inlier_threshold=inlier_threshold, show_image=show_image, K1=Kleft, D1=Dleft, K2=Kright, D2=Dright)



        # print(args)

    # print(args)

    # K1, D1 = calibrate_sequence("data/RDMcalibration/left_*.jpg")
    # K2, D2 = calibrate_sequence("data/RDMcalibration/right_*.jpg")
    #
    # ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objp, leftp, rightp, K1, D1, K2, D2, image_size)