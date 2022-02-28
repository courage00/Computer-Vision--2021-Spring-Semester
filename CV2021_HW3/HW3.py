from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
import math


def rmse(A, B):
    error = A - B
    # print(error.shape[0])
    return np.sqrt((np.dot(error, error.T))/float(error.shape[0]))


def ssd(A, B):
    error = A - B
    # print(error)
    return np.dot(error, error.T)


def compute_error(d1, d2s, math_function):
    results_index = []
    for d2 in d2s:
        results_index.append(math_function(d1, d2))
    results_index = np.argsort(results_index)
    ratio = math_function(d1, d2s[results_index[0]]) / \
        math_function(d1, d2s[results_index[1]])
    return ratio, results_index[0]


def match_feature(des1, des2, kp1, kp2, threshold=0.2):
    matches = []
    for i in range(len(des1)):
        ratio, des2_index = compute_error(des1[i], des2, ssd)
        # ratio, des2_index = compute_error(des1[i], des2,rmse)
        if ratio < threshold:
            matches.append([kp1[i].pt, kp2[des2_index].pt])
    return matches


def match_all_point(img1, img2, gray_img1, gray_img2, target_path, img_name):
    # part1 points detection & feature description
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray_img1, None)
    kp2, des2 = sift.detectAndCompute(gray_img2, None)

    # part2 Feature matching by SIFT features
    match_points = match_feature(des1, des2, kp1, kp2)

    # show drawing line image
    hA, wA, cA = img1.shape
    hB, wB, cB = img2.shape

    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")

    vis[0:hA, 0:wA] = img1
    vis[0:hB, wA:] = img2

    for (corA, corB) in match_points:
        color = np.random.randint(0, high=255, size=3).tolist()

        int_corA = (int(corA[0]), int(corA[1]))
        int_corB = (int(corB[0]) + wA, int(corB[1]))

        cv2.line(vis, int_corA, int_corB, color, 1)

    cv2.imwrite('{}{}_feature.jpg'.format(target_path, img_name), vis)
    print('\n{}{}_feature.jpg'.format(target_path, img_name))

    return match_points


def cal_homo(img1_pts, img2_pts):
    point_n = img1_pts.shape[0]
    tmep_mat = np.zeros(shape=(2*point_n, 9))

    for i in range(point_n):
        tmep_p_mat = np.zeros(shape=(2, 9))
        img2_pts_homo = np.array([img2_pts[i][0], img2_pts[i][1], 1])

        tmep_p_mat[0, 0:3] += img2_pts_homo
        tmep_p_mat[0, 6:9] += img2_pts_homo * img1_pts[i][0] * (-1)
        tmep_p_mat[1, 3:6] += img2_pts_homo
        tmep_p_mat[1, 6:9] += img2_pts_homo * img1_pts[i][1] * (-1)

        tmep_mat[2*i:(2*i+2), 0:9] += tmep_p_mat

    _, _, v_T = np.linalg.svd(tmep_mat)
    m = v_T.T[:, -1]
    H_mat = m.reshape(3, 3)
    H_mat /= H_mat[2][2]

    return H_mat


def cal_outliers(points, N_points, H_mat):
    # turn to be homogeneous
    pts1, pts2 = points[:, 0, :], points[:, 1, :]
    pts1_homo = np.hstack((pts1, np.ones((N_points, 1))))
    pts2_homo = np.hstack((pts2, np.ones((N_points, 1))))

    # estimated points
    pts1_homo_ = (H_mat @ pts2_homo.T).T
    pts1_homo_ /= pts1_homo_[:, 2].reshape(-1, 1)

    # calculate the geometry distance
    distance = np.linalg.norm((pts1_homo - pts1_homo_), axis=1, keepdims=True)
    # print(distance)
    outlier_index = np.where(distance > 5)[0]
    N_outliers = len(outlier_index)

    return N_outliers


def ransac(match_points, target_path, img_name):
    # parameters
    N_sample = 4
    N_iter = 2000

    # initial
    N_points = match_points.shape[0]
    least_N_outliers = N_points
    best_H = np.zeros((3, 3))

    # iterate for N times
    for i in range(N_iter):
        # sample S correspondences from the feature matching results
        sample_points = match_points[random.sample(range(N_points), N_sample)]
        img1_pts, img2_pts = sample_points[:, 0, :], sample_points[:, 1, :]

        # compute the homography matrix based on these sampled correspondences
        H_mat = cal_homo(img1_pts, img2_pts)

        # check the number of outliers by a threshold
        N_outliers = cal_outliers(
            match_points, N_points, H_mat)

        # get the best homography matrix with smallest number of outliers
        if N_outliers < least_N_outliers:
            best_H = H_mat
            print('Homography:', best_H)
            least_N_outliers = N_outliers
            if N_outliers == 0:
                break

    return best_H


def crop_black_boundary(img):
    if len(img.shape) == 3:
        n_rows, n_cols, c = img.shape
    else:
        n_rows, n_cols = img.shape
    row_low, row_high = 0, n_rows
    col_low, col_high = 0, n_cols

    for row in range(n_rows):
        if np.count_nonzero(img[row]) > 0:
            row_low = row
            break
    for row in range(n_rows - 1, 0, -1):
        if np.count_nonzero(img[row]) > 0:
            row_high = row
            break
    for col in range(n_cols):
        if np.count_nonzero(img[:, col]) > 0:
            col_low = col
            break
    for col in range(n_cols - 1, 0, -1):
        if np.count_nonzero(img[:, col]) > 0:
            col_high = col
            break

    return img[row_low:row_high, col_low:col_high]


def bilinear_interpolate(img, x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    h, w, c = img.shape

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    return wa[:, None] * Ia + wb[:, None] * Ib + wc[:, None] * Ic + wd[:, None] * Id


def get_left_bound(mask):
    for col in range(mask.shape[1]):
        if not np.all(mask[:, col] == [False, False, False]):
            break
    return col


def overlap(point_a, point_b):
    return not (np.array_equal(point_a, [False, False, False]) or
                np.array_equal(point_b, [False, False, False]))


class Combination():
    def __init__(self, img1, img2, homography):
        self.img1 = img1
        self.img2 = img2
        h1, w1, c = self.img1.shape
        h2, w2, c = self.img2.shape
        self.resSize = (min(h1, h2), w1+w2, c)

        self.img1_warp = self.get_warp(self.img1, None)
        self.img2_warp = self.get_warp(self.img2, homography)

        self.img1_mask = self.get_mask(self.img1_warp)
        self.img2_mask = self.get_mask(self.img2_warp)

    def get_warp(self, img, H=None):
        h, w, c = img.shape
        warp_img = np.zeros(self.resSize)

        if np.array_equal(H, None):
            warp_img[:h, :w] = img
        else:
            H_inv = np.linalg.inv(H)
            us = np.arange(warp_img.shape[1])
            vs = np.arange(warp_img.shape[0])
            us, vs = np.meshgrid(us, vs)
            uvs = np.concatenate((us.reshape(1, -1), vs.reshape(1, -1),
                                 np.ones((1, warp_img.shape[1] * warp_img.shape[0]))), axis=0)
            uvs_t = np.matmul(np.linalg.inv(H), uvs)
            uvs_t /= uvs_t[2]
            us_t = uvs_t[0]
            vs_t = uvs_t[1]
            valid = (us_t > 0) * \
                (us_t < img.shape[1]) * (vs_t > 0) * (vs_t < img.shape[0])
            uvs_t = uvs_t[:, valid]
            uvs = uvs[:, valid]
            t = bilinear_interpolate(img, uvs_t[0], uvs_t[1])
            for i in range(uvs.shape[1]):
                warp_img[int(uvs[1][i]), int(uvs[0][i])] = t[i]
        return warp_img / 255

    def get_mask(self, img):
        mask = img[not np.array_equal(img, [0, 0, 0])][0]
        mask = np.array(mask, dtype=bool)
        return mask

    def alpha_blending(self):
        blend_img = self.img1_warp + self.img2_warp
        # calculate the region of the overlapping part
        right = self.img1.shape[1]
        left = get_left_bound(self.img2_mask)
        width = right - left + 1

        for col in range(left, right + 1):
            for row in range(self.img2_warp.shape[0]):
                alpha = (col - left) / (width)
                if overlap(self.img1_mask[row, col], self.img2_mask[row, col]):
                    blend_img[row, col] = (1 - alpha) * self.img1_warp[row, col] + \
                        alpha * self.img2_warp[row, col]
        return crop_black_boundary(blend_img * 255).astype(np.uint8)


if __name__ == '__main__':

    # name
    src_path = "./data/"
    target_path = './results/'

    # mkdir target_path
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # get imgs
    # img_name = str(input("img name without 1,2 : "))
    # or
    img_name = ""
    img1 = cv2.imread(src_path+img_name+"1.jpg")  # queryImage
    img2 = cv2.imread(src_path+img_name+"2.jpg")  # trainImage

    #rgb2gray
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # part 1 & 2
    match_points = match_all_point(
        img1, img2, gray_img1, gray_img2, target_path, img_name)

    # part 3
    best_H = ransac(np.array(match_points), target_path, img_name)

    # part 4
    combination = Combination(img1, img2, best_H)
    # four type of blending algorithm
    alpha_blending_img = combination.alpha_blending()
    cv2.imwrite('{}{}_alpha_blending.jpg'.format(
        target_path, img_name), alpha_blending_img)
