# Computer Vision HW2

## Task 1 - Hybrid Image

    usage: task1.py [-h] [-f1 FILENAME1] [-f2 FILENAME2] [--dir DIR] [-c1 CUTOFF_H] [-c2 CUTOFF_L] [-s] [-m]

    optional arguments:
    -h, --help            show this help message and exit
    -f1 FILENAME1, --filename1 FILENAME1
                            specify the image1 file for high-pass
    -f2 FILENAME2, --filename2 FILENAME2
                            specify the image2 file for low-pass
    --dir DIR             specify the directory for loading image files
    -c1 CUTOFF_H, --cutoff_h CUTOFF_H
                            specify the cutoff frequency of high-pass filter
    -c2 CUTOFF_L, --cutoff_l CUTOFF_L
                            specify the cutoff frequency of low-pass filter
    -s, --fftshift        whether use fftshift to center the transform
    -m, --multiple        whether compute with multiple pairs of images or just one
