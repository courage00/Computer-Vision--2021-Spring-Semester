import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import numpy as np
import tifffile
import pylab 
import math

#skimage  resize function
from skimage.transform import resize 
from skimage import transform,data

from skimage import filters
 
"""
cathedral.jpg
emir.tif
icon.tif
lady.tif
melons.tif
monastery.jpg
nativity.jpg
onion_church.tif
three_generations.tif
tobolsk.jpg
village.tif
workshop.tif
"""

import sys
import os 
import cv2
source = "./hw2_data/task3_colorizing/"
filename = os.path.basename(sys.argv[1])
imname = source+ filename
if imname[-1]=='f':
    img=tifffile.imread(imname, cv2.IMREAD_GRAYSCALE)
else:
    img=Image.open(imname)
img=np.asarray(img)
print("originalImg size",img.shape)
 
originalImg = img
#plt.imshow(img)
#pylab.show()


w,h=img.shape 

# Cutting the white border of the original image, that could affect our alignment

img=img[int(w*0.01):int(w-w*0.02),int(h*0.04):int(h-h*0.04)]

cuttedImg = img
w,h=img.shape 

# Dividing the original picture to 3 different color source.
height=int(w/3)


height=int(w/3)  
blue_=img[0:height,:] 
green_=img[height:2*height,:] 
red_=img[2*height:3*height,:] 
 
blue = blue_
red = red_
green = green_


# Processing the TIF image.
if imname[-1]=='f':
    
    #rescale to 0.1
    img = resize(img, transform.rescale(img, 0.1).shape) 
     
    
    w,h=img.shape
    print("resize new w,h:",w,h)
    
    height=int(w/3)
     
    
# Dividing the original picture to 3 different color source.
    
sobel = filters.sobel(img) 
blue=sobel[0:height,:] 
green=sobel[height:2*height,:] 
red=sobel[2*height:3*height,:] 
        
plt.figure()
plt.imshow(blue)
pylab.show()

plt.figure()
plt.imshow(green)
pylab.show()

plt.figure()
plt.imshow(red)
pylab.show() 


    
#Calculating the normalize cross-corelation

#The Cross-correlation about a,b  
def ncc(a,b):

    #a.mean(axis=0)：compressing the column [(0,0) + (1,0)  +...(n,0)] /n 
    #Arithmetic mean for columns
    a=a-a.mean(axis=0)
    b=b-b.mean(axis=0)
    return np.sum(a*b)/np.sum(np.linalg.norm(a)*np.linalg.norm(b))


#Finding the max matching value about a and b, 
def nccAlign(a, b, t):
    max = -1
    #np.linspace use to create -t,-t+1,-t+2...t 
    ivalue=np.linspace(-t,t,2*t,dtype=int)
    jvalue=np.linspace(-t,t,2*t,dtype=int)
    for i in ivalue:
        for j in jvalue:
            #print(j)
            nccDiff = ncc(a,np.roll(b,[i,j],axis=(0,1)))
            if nccDiff > max:
                max = nccDiff
                output = [i,j]
    return output
    
    
#Alignment

segments = int(os.path.basename(sys.argv[2]))
#segments default = 10
alignGtoB = nccAlign(blue,green,segments)
alignRtoB = nccAlign(blue,red,segments)

#offset
#print("(alignGtoB, alignRtoB)",alignGtoB, alignRtoB)


if imname[-1]=='f':
    print("Processing TIF for original scale matries")
    g=np.roll(green_,[alignGtoB[0]*10,alignGtoB[1]*10],axis=(0,1))
    r=np.roll(red_,[alignRtoB[0]*10,alignRtoB[1]*10],axis=(0,1))
    coloured = (np.dstack((r,g,blue_)))
else:
    #move g 
    g=np.roll(green_,alignGtoB,axis=(0,1))
    #move r
    r=np.roll(red_,alignRtoB,axis=(0,1))
    coloured = (np.dstack((r,g,blue_))).astype(np.uint8)
    
coloured=coloured[int(coloured.shape[0]*0.05):int(coloured.shape[0]-coloured.shape[0]*0.05),int(coloured.shape[1]*0.05):int(coloured.shape[1]-coloured.shape[1]*0.05)]


if imname[-1]=='f':
    tifffile.imsave("./task3Result/"+filename, coloured)
    
else:
    coloured = Image.fromarray(coloured)
    coloured.save("./task3Result/"+filename) 
    

plt.figure()    
imgResult=Image.open("./task3Result/"+filename) 
#plt.imshow(imgResult)

title="Original"
plt.subplot(1,3,1)
img = cv2.imread(imname, 0) 
    
plt.imshow(img, cmap='gray')
plt.title(title,fontsize=18)
plt.xticks([])
plt.yticks([])

title="Cutted"
plt.subplot(1,3,2) 
plt.imshow(cuttedImg, cmap='gray')
plt.title(title,fontsize=18)
plt.xticks([])
plt.yticks([])


title="Result"
plt.subplot(1,3,3) 
plt.imshow(imgResult)
plt.title(title,fontsize=18)
plt.xticks([])
plt.yticks([])

plt.show() 
 
pylab.show()