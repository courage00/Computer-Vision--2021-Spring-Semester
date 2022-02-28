from cv2 import cv2
import os
import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image 
# 2021/4/11 Wood

# Convolve
def convolve(img, hight, width, filter):
    output_img = np.zeros(img.shape)
    for i in range(hight):
        for j in range(width):
            sum = 0.0
            for k in range(-2,3):
                for l in range(-2,3):
                    if ((i + k >= 0) and (i + k < hight)) and ((j + l >= 0 ) and (j + l < width)):
                        sum += img[(i + k)][j + l] * filter[k+2][l+2]   
            if sum > 255 :
                sum = 255                   
            output_img[i][j] = sum
    return(output_img)

def gaussian_filter(sigma):
    output_filter = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    sum = 0.0
    for i in range(5):
        for j in range(5):
            idex_x = i - 2
            idex_y = j - 2
            power = -((idex_x**2 + idex_y**2) / 2 * sigma**2)
            coefficient = 1 / (2 * np.pi * sigma**2)
            temp_val = coefficient * np.exp(power)
            output_filter[i, j] = temp_val
            sum += temp_val
    output_filter /= sum
    # sum = 0.0
    # for i in range(5):
    #     for j in range(5):
    #         sum += output_filter[i, j] 
    # print(sum)
    return output_filter

# Downsample image
def down_sample(img, ratio=2):
    
    delete_row_index=[]
    delete_column_index=[]

    R = img[:,:,0:1].copy()
    G = img[:,:,1:2].copy()
    B = img[:,:,2:].copy()
    
    count = 0
    while(count<R.shape[0]):
        delete_row_index.append(count)
        count += ratio
    R = np.delete(R, delete_row_index, axis=0)
    
    count = 0
    while(count<R.shape[1]):
        delete_column_index.append(count)
        count += ratio
    R = np.delete(R, delete_column_index, axis=1)
    
    count = 0
    while(count<G.shape[0]):
        delete_row_index.append(count)
        count += ratio
    G = np.delete(G, delete_row_index, axis=0)
    
    count = 0
    while(count<G.shape[1]):
        delete_column_index.append(count)
        count += ratio
    G = np.delete(G, delete_column_index, axis=1)
    
    count = 0
    while(count<B.shape[0]):
        delete_row_index.append(count)
        count += ratio
    B = np.delete(B, delete_row_index, axis=0)
    
    count = 0
    while(count<B.shape[1]):
        delete_column_index.append(count)
        count += ratio
    B = np.delete(B, delete_column_index, axis=1)

    return np.concatenate((R,G,B),axis=2)

def task_2(sigma):
    #建立 gaussian_filter
    g_filter = gaussian_filter (sigma)
    print(g_filter)
    #建置存放資料夾
    hybrid_path = "./hw2_data/task1,2_hybrid_pyramid/"
    hybrid_name = sorted(os.listdir(hybrid_path))
    pyramid_floder = []
    pyramid_floder_class = []

    for i in range(len(hybrid_name)):
        current_name_list = hybrid_name[i].split('.')
        if (len(current_name_list) == 2 and (current_name_list[0] !=  "")):
            pyramid_floder.append(current_name_list[0])
            pyramid_floder_class.append(current_name_list[1])

    pyramid_path = "./hw2_data/task1,2_hybrid_pyramid/pyramid/"
    for i in range(len(pyramid_floder)):
        print("waiting for " + pyramid_floder[i])
        path_dir = pyramid_path + pyramid_floder[i]
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)   
        current_img = np.array( cv2.imread(hybrid_path+pyramid_floder[i]+"."+pyramid_floder_class[i]))     
        for times in range(1,6):
            current_transpose_img = current_img.transpose(2,1,0)
            gaussian_transpose__img = np.zeros(current_transpose_img.shape)
            for j in range(3):
                gaussian_transpose__img[j] = convolve(current_transpose_img[j], current_transpose_img.shape[1], current_transpose_img.shape[2], g_filter)
            gaussian_img = gaussian_transpose__img.transpose(2, 1, 0)
            ##保留一張單純高斯濾鏡後的
            if times == 1 :
                cv2.imwrite(path_dir+"/"+pyramid_floder[i]+"_1x.jpg", gaussian_img)
                fft_data= np.fft.fft2(gaussian_img)
                fft_shift = np.fft.fftshift(fft_data) 
                fft_img = 20 * np.log(np.abs(fft_shift))
                cv2.imwrite(path_dir+"/"+pyramid_floder[i]+"_fft_1x.jpg", fft_img)   

            temp_current_img = current_img
            current_img = down_sample(gaussian_img)
            #存高斯濾過後的圖和其fft圖片
            cv2.imwrite(path_dir+"/"+pyramid_floder[i]+"_"+ str(2 ** times) +"x.jpg", current_img)
            fft_data= np.fft.fft2(current_img)
            fft_shift = np.fft.fftshift(fft_data) 
            fft_img = 20 * np.log(np.abs(fft_shift))
            cv2.imwrite(path_dir+"/"+pyramid_floder[i]+"_fft_"+ str(2 ** (times)) +"x.jpg", fft_img)

            if times != 5 :
                laplacian_img = temp_current_img - \
                     cv2.resize(current_img, (temp_current_img.shape[1],temp_current_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                #存rgb版本
                cv2.imwrite(path_dir+"/"+pyramid_floder[i]+"_laplacian_rgb_"+ str(2 ** (times -1)) +"x.jpg", laplacian_img)
                laplacian_img = (laplacian_img-np.min(laplacian_img))/(np.max(laplacian_img)-np.min(laplacian_img))*255
                # print(laplacian_img)
                laplacian_img = laplacian_img.astype('uint8')
                # print(laplacian_img)
                laplacian_gray_img = cv2.cvtColor( (laplacian_img),cv2.COLOR_BGR2GRAY) 
                #存灰階版本圖片和其fft圖片
                cv2.imwrite(path_dir+"/"+pyramid_floder[i]+"_laplacian_gray_"+ str(2 ** (times -1)) +"x.jpg", laplacian_gray_img)
                fft_data= np.fft.fft2(laplacian_gray_img)
                fft_shift = np.fft.fftshift(fft_data) 
                fft_img = 20 * np.log(np.abs(fft_shift))
                cv2.imwrite(path_dir+"/"+pyramid_floder[i]+"_laplacian_gray_fft_"+ str(2 ** (times -1)) +"x.jpg", fft_img)


sigma = float(input("gaussian_filter 's sigma : "))
task_2(sigma)
