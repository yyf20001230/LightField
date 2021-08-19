import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import scipy.misc
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import skimage

import code as code

import os


# The methods that openCV can use for correlation
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

def crop_search_window(img, window_size, template_size, focus_center = None):
    """
        This function crops the windows and the template from the center of the first frame. 
        
        The center of the template and window is defined by focus_center (e.g. 250,300). If focus_center is 
        not passed as an argument (check with if focus_center == None ) , simply calculate the center of the image
        
        HINT: You need to calculate the top-left coordinate for both the window and the template
        where you want to start cropping
        
        You then can simply return these indices from your data.
        
        E.g. you could define something like
        
        x0 = 100
        y0 = 50
        x1 = x0 + size_x
        y1 = y0 + size y
        
        tmp = data[x0:x1,y0:y1]
        
    Args:
        template_size(int): This int (e.g. 20) is the size of the template which we are cropping from the first frame of the video.
        window_size(int): Windows size (e.g. 120), should be larger than template_size
        img(ndarray): Video data with sahpe ((272, 480, 3, 51))
        focus_center(list): 2D array which is where we want to focus at. If none simply take the center of the image
        
    Returns:
        windows(ndarray): Should be (window_size,window_size,3,num_mgs)
        template(ndarray): Should be (template_size,template_size,3) - only once for initial frame
        top_left_window(list): top-left index coordinates where we are cropping the window_size e.g. [100,400]
        top_left_template(list): top-left index coorinates where the termplate is cropped e.g. [110,410]
        
    """
    
    x_center = img.shape[1]
    y_center = img.shape[0]
    
    if np.all(focus_center):
        x_center = focus_center[1] * 2
        y_center = focus_center[0] * 2

    xmin = int((x_center - window_size) / 2)
    xmax = int((x_center + window_size) / 2)
    ymin = int((y_center - window_size) / 2)
    ymax = int((y_center + window_size) / 2)
    xmin_template = int((x_center - template_size) / 2)
    xmax_template = int((x_center + template_size) / 2)
    ymin_template = int((y_center - template_size) / 2)
    ymax_template = int((y_center + template_size) / 2)
    windows = img[ymin:ymax,xmin:xmax,:,:]
    template = img[ymin_template:ymax_template,xmin_template:xmax_template,:,0]
    return windows, template, [ymin,xmin], [ymin_template,xmin_template]


        
def findCorrelation(img,template,method='cv2.TM_CCOEFF_NORMED'):
    """'
    
    Correlate the window-search image with the template.
    You can use the cv2.matchTemplate function for this and then use
    cv2.minMaxLoc to get the maximum correlation value
    
    HINT: To use cv2.matchtemplate you need to initialize method_cv = eval(method)
    and then pass method_cv into the cv2.matchTemplate function
    
    Args:
        img(ndarray): RGB image of the window
        template(ndarray): RGB image of the template (shape should be smaller than img)
        method(str): the correlation method you want to use. 
        By default we're using only cv2.TM_CCOEFF_NORMED, but feel free to test out
        different methods
        
    Returns:
        Res(ndarray): correlation image (gray scale, size defined as in cv2-documentation )
        top_left(ndarray): the location of the maximum in the correlation image cast to int (not uint8)
    
    '"""
    
    
    
    method_cv = eval(method)
    res = cv2.matchTemplate(img,template,method_cv)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
    return res,maxLoc[::-1]
    


def findCorrelationAll(imgs,template,method='cv2.TM_CCOEFF_NORMED'):
    """
    
    Correlates all images in the video stack with the template function
    
    HINT: Loop through all frames and use the findCorrelation function
    to perform the correlation.
    
    The output size of the correlated image is a different size than the input,
    hence you need to calculate it if you want to initialize your result image
    before the for-loop
    
    See documentation here:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
    
    Args:
        imgs(ndarray): RGB video of the window with shape (Nx,Ny,3,Num_imgs)
        template(ndarray): RGB image of the template (shape should be smaller than img)
        method(str): the correlation method you want to use. 
        By default we're using only cv2.TM_CCOEFF_NORMED, but feel free to test out
        different methods
        
    Returns:
        Res(ndarray): correlation video (gray scale, size defined as in cv2-documentation )
        top_left(ndarray): the location of the maximum in the correlation image  for each frame. This should be cast to int (not uint8) as np.array
        with shape (2, num_images)
    
    """
    temp,_ = findCorrelation(imgs[:,:,:,0],template)
    res = np.empty((imgs.shape[3],temp.shape[0],temp.shape[1]))
    maxLoc = np.empty((2,imgs.shape[3]))
    for i in range(imgs.shape[3]):
        res[i,:,:],maxLoc[:,i] = findCorrelation(imgs[:,:,:,i],template)
        
    return res,maxLoc.astype(int)
        
                  
          

def correct_correlation_result_to_image_coordinates(top_left_correlated_list,imgs,window,template):
    """
    
    The coordinate outputs of the openCV correlation don't match up with the original coordinates.
    
    Look up the required formula to convert the retrieved pixel locations from the video stack
    back into the original coordinate system
    
    Find more here: https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
    
    HINT: This a one liner. Essentially you only have to add a constant. But we'll
    leave it up to you to figure out which constant needs to be added!
    
    HINT2: Not all inputs need to be used :) Think about what you actually need!
    
    
    Args:
        top_left_correlated_list(ndarray): (2,Num_imgs), the location of the correlation maximum for each image
        imgs(ndarray): pointer to the imgs
        window(ndarray): pointer to the data of the window
        termplate(ndarray): pointer to the template data
    
    Returns:
        out: ndarray(int) - The corrected pixel locations in original image space
    
    """
    result = np.empty((top_left_correlated_list.shape))
    for j in range(top_left_correlated_list.shape[1]):
        result[0,j] = top_left_correlated_list[0,j] + template.shape[0] / 2
        result[1,j] = top_left_correlated_list[1,j] + template.shape[1] / 2
        
    return result.astype(int)


def calculate_pixel_shifts(top_left_correlated):
    """
    
    calculate_pixel_shifts calculate the registration shift between each frame in the video
    compared to the first frame.
    
    I.e. the pixel shift for the first frame would be (0,0) but different for alle the others.
    
    Simply substract the other shifts from the first one to get the wanted output!
    
    Args:
        top_left_correlated_list(ndarray): (2,Num_imgs), the location of the correlation maximum for each image
    
    Returns:
        pixel_shifts(ndarray): The pixel shifts between different frames
    
    """
    pixel_shifts = np.empty((top_left_correlated.shape))
    pixel_shifts[:,0] = [0,0]
    for i in range(1,pixel_shifts.shape[1]):
        pixel_shifts[:,i] = top_left_correlated[:,i] - top_left_correlated[:,0]
        
    return pixel_shifts

    
def translate_image(img,dx,dy):
    """
    
    Translates an image by dx and dy. You can use
    cv2.warpAffne for this. 
    HINT: Look at the function in part1. You can probably resuse your code!
    
    Args:
        img(ndarray): Rgb array to be shifted
        dx(float): amount to be shifted in x-direction
        dy(float): amount to be shifted in y-direction
        
    Returns:
        res(ndarray): translated image, same shape as input.
    
    """
    M = np.float64([[1, 0, dx],[0, 1, dy]])
    result = cv2.warpAffine(img,M=M,dsize=(img.shape[1], img.shape[0]))
    return result


def translate_all_images(imgs,pixel_shifts):
    """
    
    Translates all images in the vido given the amount
    defined by pixel_shifts that you retrieve from
    calculate_pixel_shifts
    
    Args:
        imgs(ndarray): RGB-video array
        pixel_shifts(ndarray): pixel shifts (2,Num_x)
    
    Returns:
        out(ndarray): shifted images with same shape as imgs
    
    """
    shifted_imgs = np.empty((imgs.shape))
    for i in range(imgs.shape[3]):
        shifted_imgs[:,:,:,i] = translate_image(imgs[:,:,:,i],-pixel_shifts[1,i],-pixel_shifts[0,i])
        
    return shifted_imgs.astype(np.uint8)
        


def average_images(imgs):
    """
    
    Averages the image stack along the temporal dimension.
    This is a one-liner. Do not USE a for-loop for this.
    
    Args:
        imgs(ndarray): RGB array with several frames
        
    Returns:
        out(ndarray): average RGB image over the temporal dimensions
    
    """
    
    return np.mean(imgs,axis = 3).astype(np.uint8)



