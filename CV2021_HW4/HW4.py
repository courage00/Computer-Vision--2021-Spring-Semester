import numpy as np
import cv2
import random
import os
from matplotlib import pyplot as plt


def ssd(A, B):
    error = A - B
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
        if ratio < threshold:
            matches.append([kp1[i].pt, kp2[des2_index].pt])
    return matches


def match_all_point(img1, img2, target_path, img_name, threshold=0.2):
    # Points detection & feature description
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Feature matching by SIFT features
    match_points = match_feature(des1, des2, kp1, kp2, threshold)

    # Show drawing line image
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

    cv2.imwrite(os.path.join(target_path, f'{img_name}_feature.jpg'), vis)

    return match_points


def homogeneous(points):
    if points.shape[1] == 3:
        return points

    ones = np.ones(points.shape[0])
    homo_points = np.vstack((points.T, ones))

    return homo_points.T


def plot_epilines(img1, img2, inlier_mask, match_points, target_path, img_name):
    hA, wA = img1.shape[:2]
    hB, wB = img2.shape[:2]

    inlier_mask = np.array(inlier_mask)
    in_pts1 = match_points[:, 0][inlier_mask.ravel() == 1]
    in_pts2 = match_points[:, 1][inlier_mask.ravel() == 1]

    pts2_homo = homogeneous(in_pts2)
    line2 = np.dot(pts2_homo, best_F)

    img1_tmp = img1.copy()
    img2_tmp = img2.copy()

    # plot epipolar lines
    for (corA, corB, L2) in zip(in_pts1, in_pts2, line2):
        color = np.random.randint(0, high=255, size=3).tolist()

        x0, y0 = map(int, [0, -L2[2]/L2[1]])
        x1, y1 = map(int, [wB, -(L2[2]+L2[0]*wB)/L2[1]])
        img1_tmp = cv2.line(img1_tmp, (x0, y0), (x1, y1), color, 1)
        img2_tmp = cv2.circle(img2_tmp, tuple(
            map(int, corB[:2])), 5, color, -1)
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")

    vis[0:hA, 0:wA] = img1_tmp
    vis[0:hB, wA:] = img2_tmp
    cv2.imwrite(os.path.join(
        target_path, f'{img_name}_epipolar_lines.jpg'), vis)


def cal_valid_point(project, X):
    r = project[:, 0:3]
    t = project[:, 3]
    count = 0
    for target in X:
        # left z-axis dot target in left camera coordination
        left_check = np.dot(target, np.array([0, 0, 1]))
        # right z-axis dot target in right camera coordination
        right_check = np.dot((target - t), np.expand_dims(r[-1, :], axis=1))
        if left_check > 0 and right_check > 0:
            count += 1
    return count


def out_csv(data, data_name):
    with open(data_name, 'w') as fp:
        for row in data:
            row_join = ','.join(map(str, row))
            fp.write(row_join)
            fp.write('\n')
    print(data_name)


class Fundamental():
    def __init__(self, shape1, shape2, match_points):
        self.img1_shape = shape1
        self.img2_shape = shape2
        self.match_points = match_points

    def normalize(self, img_shape, pts):
        """Normalize image coordinates."""
        h, w = img_shape[0], img_shape[1]
        ones = np.ones(pts.shape[0]).reshape(-1, 1)
        pts = np.concatenate((pts, ones), axis=1).T
        norm_mat = np.array([[2/w, 0,  -1],
                             [0, 2/h, -1],
                             [0,  0,   1]])
        normalized_pts = norm_mat @ pts

        return normalized_pts, norm_mat

    def fundamental_mat(self, img1_pts, img2_pts):
        mat = []
        points_num = img1_pts.shape[1]
        for i in range(points_num):
            x1 = img1_pts[0][i]
            y1 = img1_pts[1][i]
            x2 = img2_pts[0][i]
            y2 = img2_pts[1][i]

            mat.append([x2*x1, x2*y1, x2,
                        y2*x1, y2*y1, y2,
                        x1, y1, 1])
        mat = np.array(mat)
        _, _, V = np.linalg.svd(mat)
        F = V[-1].reshape(3, 3)
        U, S, V = np.linalg.svd(F)
        S[2] = 0
        F = U @ np.diag(S) @ V
        return F

    def cal_inliers(self, pts1, pts2, F_mat, threshold):
        # sampson distance
        L1 = F_mat.T @ pts1
        L2 = F_mat @ pts2

        JJT = L1[0]**2 + L1[1]**2 + L2[0]**2 + L2[1]**2
        err = np.diag(pts2.T@F_mat@pts1)**2 / JJT

        num_inlier = 0
        inliers_mask = []
        # Compute num of inliners
        for j in range(len(err)):
            if err[j] <= threshold:
                num_inlier += 1
                inliers_mask.append(1)
            else:
                inliers_mask.append(0)
        return num_inlier, inliers_mask

    def ransac_and_denormalize(self, iters=1000, threshold=1e-4):
        """Compute best fundamental matrix by RANSAC with 8-points algorithm."""
        pts1, mat1 = self.normalize(self.img1_shape, self.match_points[:, 0])
        pts2, mat2 = self.normalize(self.img2_shape, self.match_points[:, 1])

        points_num = pts1.shape[1]
        last_inliers_num = 0
        best_F = np.zeros((3, 3))
        best_inliers_mask = []

        for i in range(iters):
            # sample 8 correspondences from the feature matching results
            idx = random.sample(range(points_num), 8)
            sample_pts1 = pts1[:, idx]
            sample_pts2 = pts2[:, idx]

            # compute the fundamental matrix based on these sampled correspondences
            F_mat = self.fundamental_mat(sample_pts1, sample_pts2)

            # check the number of inliers by a threshold
            inliers_num, inliers_mask = self.cal_inliers(
                pts1, pts2, F_mat, threshold)

            if inliers_num > last_inliers_num:
                # Denormalize
                best_F = mat2.T @ F_mat @ mat1
                best_inliers_mask = inliers_mask
                print(f'inliers_num: {inliers_num}, iteration: {i}')
                if inliers_num == points_num:
                    break
                last_inliers_num = inliers_num

        return best_F/best_F[-1, -1], best_inliers_mask


class Essential():
    def __init__(self, img_name):
        self.name = img_name

    def get_K(self):
        if self.name.find('Statue') != -1:
            K1 = np.array([[5426.566895,    0.678017, 330.096680],
                           [0.000000, 5423.133301, 648.950012],
                           [0.000000,    0.000000,   1.000000]])
            K2 = np.array([[5426.566895,    0.678017, 387.430023],
                           [0.000000, 5423.133301, 620.616699],
                           [0.000000,    0.000000,   1.000000]])
        elif self.name.find('Mesona') != -1:
            K1 = np.array([[1.4219, 0.0005, 0.5092],
                           [0.0000, 1.4219, 0.3802],
                           [0.0000, 0.0000, 0.0010]])
            K2 = K1
        elif self.name.find('fish') != -1:
            K1 = np.array([[1.32637928e+03, -1.79134895e+00, 7.53924281e+02],
                           [0.00000000e+00,  1.32632297e+03, 5.83461910e+02],
                           [0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
            K2 = K1

        self.K1 = K1
        self.K2 = K2

    def essential_mat(self, F):
        self.get_K()
        self.E = np.dot(np.dot(self.K1.T, F), self.K2)

        return self.E

    def compute_possible_P2(self):
        """Compute four possible choices for the second camera matrix
            from an essential matrix E = [t]R
        """
        U, _, V = np.linalg.svd(self.E)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        # Ensure rotation matrix are right-handed with positive determinant
        if np.linalg.det(np.dot(U, V)) < 0:
            V = -V

        r1 = np.dot(np.dot(U, W), V.T)
        r2 = np.dot(np.dot(U, W.T), V.T)

        u3 = np.expand_dims(U[:, 2], axis=1)

        P2s = [
            np.hstack((r1, u3)),
            np.hstack((r1, -u3)),
            np.hstack((r2, u3)),
            np.hstack((r2, -u3))
        ]

        return P2s


if __name__ == '__main__':
    # img_name, img_type = 'Mesona', 'jpg'
    img_name, img_type = 'Statue', 'bmp'
    # img_name, img_type = 'fish', 'jpg'
    src_path, target_path = 'data', 'results'
    threshold = 0.2

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    img1 = cv2.imread(os.path.join(
        src_path, f'{img_name}1.{img_type}'))  # queryImage
    img2 = cv2.imread(os.path.join(
        src_path, f'{img_name}2.{img_type}'))  # trainImage

    # 1. Find out correspondence across images
    match_points = np.array(match_all_point(img1, img2, target_path, img_name))
    # match_points = np.array(match_all_point(img1, img2, target_path, img_name, threshold))

    # 2. Estimate the fundamental matrix across images (normalized 8 points)
    fund = Fundamental(img1.shape[:2], img2.shape[:2], match_points)
    best_F, inlier_mask = fund.ransac_and_denormalize()
    print(f'Fundamental matrix:\n{best_F}')

    # 3. Draw the interest points on you found in step.1 in one image
    #    and the corresponding epipolar lines in another
    plot_epilines(img1, img2, np.array(inlier_mask),
                  match_points, target_path, img_name)

    # 4. Get 4 possible solutions of essential matrix from fundamental matrix
    essential = Essential(img_name)
    E = essential.essential_mat(best_F)
    print(f'Essential matrix:\n{E}')

    P1 = np.eye(3, 4)
    P2s = essential.compute_possible_P2()

    # 5. Find out the most appropriate solution of essential matrix and apply
    #    triangulation to get 3D points
    proj1 = np.dot(essential.K1, P1)
    proj2s = []
    for p2 in P2s:
        proj2s.append(np.dot(essential.K2, p2))
    proj2s = np.array(proj2s)

    max_point = 0
    for proj2 in proj2s:
        X = []
        inlier_mask = np.array(inlier_mask)
        in_pts1 = match_points[:, 0][inlier_mask.ravel() == 1]
        in_pts2 = match_points[:, 1][inlier_mask.ravel() == 1]
        for (u1, v1), (u2, v2) in zip(in_pts1, in_pts2):
            A = []
            A.append(v1 * proj1[2, :] - proj1[1, :])
            A.append(u1 * proj1[1, :] - v1 * proj1[0, :])
            A.append(v2 * proj2[2, :] - proj2[1, :])
            A.append(u2 * proj2[1, :] - v2 * proj2[0, :])
            A = np.array(A)
            A_U, A_S, A_V = np.linalg.svd(A)
            last_V = A_V[np.argmin(A_S)]
            X.append((last_V/last_V[3])[:3])

        valid_point_num = cal_valid_point(proj2, X)
        if valid_point_num > max_point:
            max_point = valid_point_num
            correct_X = X

    correct_X = np.asarray(correct_X)
    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.gca(projection='3d')
    ax.scatter(correct_X[:, 0], correct_X[:, 1], correct_X[:, 2], c=correct_X[:, 2], cmap='gist_rainbow_r', marker='^')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    plt.show()

    # output csv
    target_path = os.path.join(target_path, img_name)
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    out_csv(proj1, os.path.join(target_path, 'CameraMatrix.csv'))
    out_csv(match_points[:, 0][inlier_mask.ravel() == 1],
            os.path.join(target_path, 'pts2D.csv'))
    out_csv(correct_X, os.path.join(target_path, 'pts3D.csv'))
