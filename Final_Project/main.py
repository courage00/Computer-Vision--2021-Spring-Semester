#!/usr/bin/env python3

import math
import cv2
import dlib
import numpy as np
import sys
import argparse
import logging
import os

predictor_path = "resources/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
COLOUR_CORRECT_BLUR = 0.5

MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))

POINTS = LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + \
    RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS
ALIGN_POINTS = POINTS
OVERLAY_POINTS = [POINTS]


def get_landmarks(img):
    rects = detector(img, 1)
    if len(rects) == 0:
        return -1

    return np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im, landmarks[group], color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0
    im = im * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def transformation_f_points(points1, points2):
    # Cast all matrix inputs into 64-bit floating point matrices.
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    # Determine the centroid for all landmark points.
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)

    # Subtract the centroids from the relative points per image.
    points1 -= c1
    points2 -= c2

    # Determine the standard deviation to remove the scaling issue
    # from the e Orthogonal Procrutes Problem.
    s1 = np.std(points1)
    s2 = np.std(points2)

    points1 /= s1
    points2 /= s2

    # Rotate the corresponding image B to A.
    u, s, vt = np.linalg.svd(points1.T * points2)
    r = (u*vt).T

    # Determine and return an affine transformation matrix.
    h_stack = np.hstack(((s2/s1) * r, c2.T - (s2/s1) * r * c1.T))
    return np.vstack([h_stack, np.matrix([0., 0., 1.])])


def warp_im(im, m, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im, m[:2], (dshape[1], dshape[0]), dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colours(im1, im2, landmarks1):
    mean_left = np.mean(landmarks1[LEFT_EYE_POINTS], axis=0)
    mean_right = np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0)

    blur_amount = COLOUR_CORRECT_BLUR * np.linalg.norm(mean_left - mean_right)
    blur_amount = int(blur_amount)

    if blur_amount % 2 == 0:
        blur_amount += 1

    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


def face_swap_filter(swap_img, swap_img_landmarks, src_img, mask_img):
    me_img = cv2.imread(os.path.join("data", src_img), cv2.IMREAD_COLOR)
    me_img = cv2.resize(
        me_img, (me_img.shape[1] * SCALE_FACTOR, me_img.shape[0] * SCALE_FACTOR))
    me_landmarks = get_landmarks(me_img)

    # Plot the landmarks on source and mask images
    im = annotate_landmarks(me_img, me_landmarks)
    cv2.imwrite(f"landmarks_{src_img}.png", im)
    sw = annotate_landmarks(swap_img, swap_img_landmarks)
    cv2.imwrite(f"landmarks_{mask_img}.png", sw)

    if type(me_landmarks) is not int:
        # Compute the transformation matrix.
        m = transformation_f_points(
            me_landmarks[ALIGN_POINTS], swap_img_landmarks[ALIGN_POINTS])

        # Colour correcting the mask image
        warped_swap = warp_im(swap_img, m, me_img.shape)
        warped_corrected_swap = correct_colours(
            me_img, warped_swap, me_landmarks)

        # Generate the blending mask
        mask = get_face_mask(swap_img, swap_img_landmarks)
        warped_mask = warp_im(mask, m, me_img.shape)
        combined_mask = np.max(
            [get_face_mask(me_img, me_landmarks), warped_mask], axis=0)

        output_im = me_img * (1.0 - combined_mask) + \
            warped_corrected_swap * combined_mask
        name = f"swap_{src_img}_{mask_img}.png"
        cv2.imwrite(name, output_im)
        out = cv2.imread(name, 1)
        cv2.imshow("Swap Output", out)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--source", type=str, default="2.jpg")
    argparser.add_argument("-mf", "--mask_folder",
                           type=str, default="morph_result")
    argparser.add_argument("-m", "--mask", type=str, default="frame010.png")
    args = argparser.parse_args()

    mask = cv2.imread(os.path.join(
        args.mask_folder, args.mask), cv2.IMREAD_COLOR)
    swap_img_landmarks = get_landmarks(mask)

    try:
        while True:
            face_swap_filter(mask, swap_img_landmarks, args.source, args.mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass