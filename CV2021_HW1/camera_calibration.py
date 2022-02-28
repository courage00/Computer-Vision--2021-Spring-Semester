import numpy as np
from cv2 import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints_ur = [] # 3d points in real world space. (unrotated)
imgpoints_ur = [] # 2d points in image plane. (unrotated)
objpoints_r = [] # 3d points in real world space. (may be rotated)
imgpoints_r = [] # 2d points in image plane. (may be rotated)

# Make a list of calibration images
images = glob.glob('data/*.jpg')
image_rotation = []
image_num = 10

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use aspect ratio of the first image as benchmark
    if idx == 0:
        fix_size = gray.shape

    # Find the chessboard corners
    print('find the chessboard corners of', fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints_ur.append(objp)
        imgpoints_ur.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)
        if (gray.shape == fix_size):
            objpoints_r.append(objp)
            imgpoints_r.append(corners)
        else:            
            gray = np.rot90(gray)
            image_rotation.append(idx)
            ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)
            if ret == True:
                objpoints_r.append(objp)
                imgpoints_r.append(corners)
            else:
                print('Fail to find the corners after rotating：', fname)


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
# img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
# Vr = np.array(rvecs)
# Tr = np.array(tvecs)
# extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)


def calc_Hi(objpoints, imgpoints, image_num, point_num):    
    H_list = []
    H_coefficient = np.zeros((point_num*2, 9))
    
    for i in range(image_num):
        objp_temp = objpoints[i].copy()
        objp_temp[:, 2] = 1
        imgpoints_temp = np.squeeze(imgpoints[i].copy())
        
        for num in range(point_num):
            H_coefficient[num*2, :] = [objp_temp[num, 0],objp_temp[num, 1], 1, 0, 0, 0, -(objp_temp[num, 0] * imgpoints_temp[num, 0]),\
                          -(objp_temp[num, 1] * imgpoints_temp[num, 0]), -(imgpoints_temp[num, 0])]
            H_coefficient[num*2 + 1, :] = [0, 0, 0, objp_temp[num, 0],objp_temp[num, 1], 1, -(objp_temp[num, 0] * imgpoints_temp[num, 1]),\
                          -(objp_temp[num, 1] * imgpoints_temp[num, 1]), -(imgpoints_temp[num, 1])]
        
        H = np.linalg.svd(H_coefficient, full_matrices=False)[2].T[:, -1] 
        if H[-1] < 0:
            H = -H
        
        H = H.reshape((3, 3))                    
        H_list.append(H)
        
    return H_list

def calc_B(H_list, image_num):
    B = np.zeros((3, 3))
    B_coefficient = np.zeros((image_num*2, 6))
    
    for i in range(image_num):
        H = H_list[i]
        h_1 = H[:, 0]
        h_2 = H[:, 1]
        
        B_coefficient[2*i][0] = h_1[0]*h_2[0]
        B_coefficient[2*i][1] = h_1[0]*h_2[1] + h_1[1]*h_2[0]
        B_coefficient[2*i][2] = h_1[0]*h_2[2] + h_1[2]*h_2[0]
        B_coefficient[2*i][3] = h_1[1]*h_2[1]
        B_coefficient[2*i][4] = h_1[1]*h_2[2] + h_1[2]*h_2[1]
        B_coefficient[2*i][5] = h_1[2]*h_2[2]
        
        B_coefficient[2*i + 1][0] = h_1[0]*h_1[0] - h_2[0]*h_2[0]
        B_coefficient[2*i + 1][1] = h_1[0]*h_1[1] + h_1[1]*h_1[0] - h_2[0]*h_2[1] - h_2[1]*h_2[0]
        B_coefficient[2*i + 1][2] = h_1[0]*h_1[2] + h_1[2]*h_1[0] - h_2[0]*h_2[2] - h_2[2]*h_2[0]
        B_coefficient[2*i + 1][3] = h_1[1]*h_1[1] - h_2[1]*h_2[1]
        B_coefficient[2*i + 1][4] = h_1[1]*h_1[2] + h_1[2]*h_1[1] - h_2[1]*h_2[2] - h_2[2]*h_2[1]
        B_coefficient[2*i + 1][5] = h_1[2]*h_1[2] - h_2[2]*h_2[2]
        
    B_e = np.linalg.svd(B_coefficient, full_matrices=False)[2].T[:, -1]

    if (B_e[0] < 0 or B_e[3] < 0 or B_e[5] < 0):
        B_e = -B_e
    
    B[0][0] = B_e[0]
    B[0][1] = B_e[1]
    B[1][0] = B_e[1]
    B[1][1] = B_e[3]
    B[0][2] = B_e[2]
    B[2][0] = B_e[2]
    B[1][2] = B_e[4]
    B[2][1] = B_e[4]
    B[2][2] = B_e[5]
    
    return(B)

def calc_intrinsic(B):    
    K = np.linalg.inv(np.linalg.cholesky(B).T)  
    K = K / K[2, 2]
    
    return(K)

def calc_extrinsic(H_list, K_rotate_inv, K_unrotate_inv, image_num):
    extrinsic_list = np.zeros((10, 6))
    
    for i in range(image_num):
    
        rotation = False
        
        ex_r = np.zeros((3, 3))
        H_current = H_list[i]
        
        h_1 = H_current[:, 0]
        h_2 = H_current[:, 1]
        h_3 = H_current[:, 2]
        
        for j in image_rotation:
            if i == j:
                rotation = True
                break
            else:
                pass
            
        if rotation:
            lamda = np.dot(K_rotate_inv, h_1)
            lamda = (lamda[0]**2 + lamda[1]**2 + lamda[2]**2)**(0.5)
            lamda = 1 / lamda
            
            r_1 = np.dot(lamda*K_rotate_inv, h_1)
            r_2 = np.dot(lamda*K_rotate_inv, h_2)
            r_3 = np.cross(r_1, r_2)
            t = np.dot(lamda*K_rotate_inv, h_3)
            
            ex_r[:, 0] = r_1
            ex_r[:, 1] = r_2
            ex_r[:, 2] = r_3
            
            ex_r = np.squeeze(cv2.Rodrigues(ex_r)[0])
            extrinsic = np.concatenate((ex_r, t), axis=0)
            extrinsic_list[i, :] = extrinsic
            
        else:
            lamda = np.dot(K_unrotate_inv,h_1)
            lamda = (lamda[0]**2 + lamda[1]**2 + lamda[2]**2)**(0.5)
            lamda = 1/lamda
            
            r_1 = np.dot(lamda*K_unrotate_inv, h_1)
            r_2 = np.dot(lamda*K_unrotate_inv, h_2)
            r_3 = np.cross(r_1,r_2)
            t = np.dot(lamda*K_unrotate_inv, h_3)
            
            ex_r[:, 0] = r_1
            ex_r[:, 1] = r_2
            ex_r[:, 2] = r_3
            
            ex_r = np.squeeze(cv2.Rodrigues(ex_r)[0])
            extrinsic = np.concatenate((ex_r, t), axis=0)
            extrinsic_list[i, :] = extrinsic
            
    return extrinsic_list


point_num = corner_x * corner_y
    
H_list_origin = calc_Hi(objpoints_ur, imgpoints_ur, image_num, point_num)
H_list = calc_Hi(objpoints_r, imgpoints_r, image_num, point_num)
B = calc_B(H_list, image_num)
K = calc_intrinsic(B)

K_unrotate = K.copy()
K_rotate = K.copy()
K_rotate[0, 0], K_rotate[1, 1] = K_rotate[1, 1], K_rotate[0, 0]
K_rotate[0, 2], K_rotate[1, 2] = K_rotate[1, 2], K_rotate[0, 2]

K_unrotate_inv = np.linalg.inv(K_unrotate)
K_rotate_inv = np.linalg.inv(K_rotate)

extrinsic_list = calc_extrinsic(H_list_origin, K_rotate_inv, K_unrotate_inv, image_num)

print('intrinsic matrix K:')
print(K)


# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = K
cam_width = 0.064 / 0.1
cam_height = 0.032 / 0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsic_list, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
