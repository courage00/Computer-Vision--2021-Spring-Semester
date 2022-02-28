file = './results/Statue/';  % 'Mesona', 'Statue' ,'fish'
src_path = './data/';

pts3D = csvread([file 'pts3D.csv']);
pts2D = csvread([file 'pts2D.csv']);
CameraMatrix = csvread([file 'CameraMatrix.csv']);
obj_main(pts3D, pts2D, CameraMatrix, [src_path 'Statue1.bmp'], 1) % 'Mesona1.JPG', 'Statue1.bmp','fish1.jpg'