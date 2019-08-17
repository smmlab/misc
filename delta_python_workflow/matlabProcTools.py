import scipy.io as sio
from skimage import io
from skimage import morphology
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import glob
import math
import scipy.ndimage as ndimage
import matplotlib as mpl
import os

def display_image_in_actual_size(im_data,cm='viridis'):

    dpi = mpl.rcParams['figure.dpi']
    height, width = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap=cm)

    plt.show()
    
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx]

def bwthicken(bw,N):
    bw = morphology.thin(-bw+1,N)
    bw = bwmorphDiag(bw)
    bw = -bw.astype(np.uint8) + 1
    bw[0:N,:]=0
    bw[-N:,:]=0
    bw[:,0:N]=0
    bw[:,-N:]=0
    
    return bw
    
def bwmorphDiag(bw):
    # filter for 8-connectivity of the background
    f = np.array(([1, -1, 0],[-1, 1, 0],[0, 0, 0]),dtype = np.int)
    # initialize result with original image
    bw = bw.astype(np.int)
    res2 = bw.copy().astype(np.bool)
    for ii in range(4): # all orientations
        # add results where sum equals 2 -> two background pixels on the
        # diagonal with 2 foreground pixels on the crossing mini-anti-diagonal
        res2 = res2 | (ndimage.filters.convolve(np.invert(bw),f) == 2)
        f = np.rot90(f) # rotate filter to next orientation
    return res2

def weightmap(seg,weight_c0,weight_c1,sigma=2,w0=12,rescale=25,zeroContour=1):
    
    [N,L] = cv2.connectedComponents(seg,0)
    N = N-1

    dist_arr = np.inf*np.ones([seg.shape[0],seg.shape[1], max(N,2)])

    for i in range(N):
        singlecell = L==i+1
        singlecell = singlecell.astype(np.uint8)
        singlecell = 1 - singlecell
        dist_arr[:,:,i] = cv2.distanceTransform(singlecell,cv2.DIST_L2, 3) 

    dist_arr = np.sort(dist_arr,axis=2)
    dist_arr_sum = dist_arr[:,:,0] + dist_arr[:,:,1]
    weight_dist = w0 * np.exp(-np.power(dist_arr_sum,2)/(2*sigma**2))

    #weight_dist = weight_dist .* np.float32(~seg)  #converts seg into single-precision float 
    weight_dist[np.nonzero(seg)]=0

    #sum up all the weights _ i definitely do not understand what's this bit supposed to do....
    weight_map = np.zeros([seg.shape[0],seg.shape[1]]) #???? do i want this?

    #weight_map(seg) = weight_c1 #can't do this w python
    weight_map[np.nonzero(seg)]=weight_c1

    #weight_map(~seg) = weight_c0 #can't do this w python --- 'can't assign to function call' !
    weight_map[np.nonzero(seg==0)]=weight_c0

    weight_map = weight_map + weight_dist

    if zeroContour:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        imsubtract = cv2.subtract(seg,cv2.erode(seg,kernel))
        weight_map[np.nonzero(imsubtract)]=0
    
    weight_map = rescale*weight_map

    return weight_map.astype(np.uint8)

def ensure_dir(new_folder):
    
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, new_folder)
    if not os.path.exists(final_directory):
       os.makedirs(final_directory)