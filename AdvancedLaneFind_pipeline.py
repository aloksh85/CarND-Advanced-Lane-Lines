#!/bin/python

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


def getCameraCalibrationMatrix(distortedImgDir, nx=9,ny=6):
    """
    Return camera calibration matrix and distortion measure
    Arguments:
    distortedImgDir : path to directory containing distorted chessboard images  
    nx : number of corners in chessboard image along x
    ny : number of corners in chessboard image along y
    """
    objpoints =[]
    imgpoints =[]
    images = glob.glob(distortedImgDir+'/calibration*.jpg')

    objp = np.zeros((ny*nx,3),np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    count = 0
    for fname in images:
        print("Processing image: ",count)
        distortedImg = cv2.imread(fname)
        gray =  cv2.cvtColor(distortedImg,cv2.COLOR_BGR2GRAY)
        ret_corners,corners = cv2.findChessboardCorners(gray,(nx,ny),None)
        count+=1
        if ret_corners:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    ret_mtx,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
    
    return mtx,dist

def displayDistortionCorrectedImgs(distortedImgDir,mtx,dist):

    images = glob.glob(distortedImgDir+'/calibration*.jpg')
    num_imgs =5
    f,axarr = plt.subplots(num_imgs,2,figsize=(15,15))

    for i,plt_num in zip(range(10,10+num_imgs),range(0,num_imgs)):
       img = cv2.imread(images[i])
       undist_img = cv2.undistort(img,mtx,dist,None,mtx)
       axarr[plt_num,0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
       axarr[plt_num,1].imshow(cv2.cvtColor(undist_img,cv2.COLOR_BGR2RGB))

    plt.show()

def thresholdImage(img,sx_thresh=(20,100),s_thresh=(150,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    # Plotting thresholded images
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    #ax1.set_title('Stacked thresholds')
    #ax1.imshow(color_binary)
        
    #ax2.set_title('Combined S channel and gradient thresholds')
    #ax2.imshow(combined_binary, cmap='gray')

    return combined_binary, color_binary

def getPerspectiveTransform(src_pts,dest_pts):
    '''
    Returns a perspective tranform that will warp an image to top view
    src_pts: an array of 4 points in input img space
    dst_pts: an array of 4 points corresponding to 'src_pts' that form a rectangle in dst_img space
    '''
    M = None
    M = cv2.getPerspectiveTransform(src_pts,dest_pts)
    return M

def warpImage(img, M):
    '''
    Return a warped img using the provided transform
    img: input image
    M: transform
    '''
    img_size=(img.shape[1],img.shape[0])
    warped_img = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
    return warped_img


def binaryImgHistogram(bin_img,x_start_stop=(0,0),y_start_stop=(0,0)):

    bin_hist = np.sum(bin_img[bin_img.shape[0]//2:,:],axis=0)
    return bin_hist


def slidingWindowSearch(binary_warped,bin_hist,nwindows=9,win_margin=100, recenter_minpix = 50):

    out_img = np.copy(binary_warped)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(bin_hist.shape[0]/2)
    leftx_base = np.argmax(bin_hist[:midpoint])
    rightx_base = np.argmax(bin_hist[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = win_margin
    # Set minimum number of pixels found to recenter window
    minpix = recenter_minpix
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        #(0,255,0), 2) 
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        #(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit


def laneLineCurvature(left_fitx,right_fitx,ploty,image_size,xm_per_pix,ym_per_pix):

   left_fit_cr = np.polyfit(ploty*ym_per_pix,left_fitx*xm_per_pix,2) 
   right_fit_cr = np.polyfit(ploty*ym_per_pix,right_fitx*xm_per_pix,2) 

   lcurve = np.power((1+((2*left_fit_cr[0]*image_size[1]*ym_per_pix)+left_fit_cr[1])**2),1.5)/np.abs(2*left_fit_cr[0])
   rcurve = np.power((1+((2*right_fit_cr[0]*image_size[1]*ym_per_pix)+right_fit_cr[1])**2),1.5)/np.abs(2*right_fit_cr[0])
   return lcurve,rcurve

def drawOnPerspectiveImage(warped,left_fitx,right_fitx,ploty,undist,Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result

def get_offset(y_pts,x_left,x_right,x_m_px,y_m_px,img_size):
    lane_center_px = int(np.mean(x_right[-1:-3] - x_left[-1:-3]))
    offset_px = int(img_size[0]*0.5) - lane_center_px
    offset_m = offset_px*x_m_px


    return offset_m


def main():

   calibration_img_dir="/home/alok/Documents/udacity_nd/CarND-Advanced-Lane-Lines/camera_cal"

   mtx,dist = getCameraCalibrationMatrix(calibration_img_dir) 
   print('cal matrix:' )
   print(mtx)
   images = glob.glob("/home/alok/Documents/udacity_nd/CarND-Advanced-Lane-Lines/test_images/*.jpg")
    
   # perspective transform
   img_size=cv2.imread(images[0]).shape
   vertices_mask = np.float32([[150,img_size[0]-20],[580,450],[700,450],[1150,img_size[0]-20]],)
   dest_points = np.float32([[250,img_size[0]],[250,30],[1000,30],[1000,img_size[0]]])
   M = getPerspectiveTransform(vertices_mask,dest_points)
   Minv = getPerspectiveTransform(dest_points,vertices_mask) 

   ym_per_pix = 30/720 # meters per pixel in y dimension
   xm_per_pix = 3.7/700 # meters per pixel in x dimension 
   
   for img_path in images:
       test_img = cv2.imread(img_path)
       undist_img = cv2.undistort(test_img,mtx,dist,None,mtx)
       #f1,arr1 = plt.subplots(1,2,figsize=(15,15))
       #arr1[0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
       #arr1[1].imshow(cv2.cvtColor(undist_img,cv2.COLOR_BGR2RGB))
       #displayDistortionCorrectedImgs(calibration_img_dir,mtx,dist)
      
       # combined threshold image
       bin_threshold_img,color_bin_img = thresholdImage(undist_img)    
       f2,arr2 = plt.subplots(3,2,figsize=(15,15))
       
       # draw mask 
       #cv2.line(undist_img, (vertices_mask[0,0],vertices_mask[0,1]),
       #  (vertices_mask[1,0],vertices_mask[1,1]), [0,0,255], 2)
       #cv2.line(undist_img, (vertices_mask[1,0],vertices_mask[1,1]),
       #  (vertices_mask[2,0],vertices_mask[2,1]), [0,0,255], 2)
       #cv2.line(undist_img, (vertices_mask[2,0],vertices_mask[2,1]),
       #  (vertices_mask[3,0],vertices_mask[3,1]), [0,0,255], 2)
       #cv2.line(undist_img, (vertices_mask[3,0],vertices_mask[3,1]),
       #  (vertices_mask[0,0],vertices_mask[0,1]), [0,0,255], 2)       
      
       warped_img = warpImage(undist_img,M)
       binary_warped_img = warpImage(bin_threshold_img,M)
       binary_histogram = binaryImgHistogram(binary_warped_img)
       left_lane_fit, right_lane_fit = slidingWindowSearch(binary_warped_img,binary_histogram)
       
       # Generate x and y values for plotting
       ploty = np.linspace(0, binary_warped_img.shape[0]-1, binary_warped_img.shape[0] )
       left_fitx = left_lane_fit[0]*ploty**2 + left_lane_fit[1]*ploty + left_lane_fit[2]
       right_fitx = right_lane_fit[0]*ploty**2 + right_lane_fit[1]*ploty + right_lane_fit[2]       
       # Generate image to plot lane lines on
       lcurve, rcurve = laneLineCurvature(left_fitx,right_fitx,ploty,(undist_img.shape[1],undist_img.shape[0]),xm_per_pix,ym_per_pix)
       offset_m = get_offset(ploty,left_fitx,right_fitx,xm_per_pix,ym_per_pix,(undist_img.shape[1],undist_img.shape[0]))
       print('left lane curvature: ',lcurve)
       print('right lane curvature: ',rcurve)
       print('offset_m: ',offset_m)
       lane_plot_img = np.dstack((binary_warped_img,binary_warped_img,binary_warped_img))*255       
       lane_drawn_img = drawOnPerspectiveImage(binary_warped_img,left_fitx,right_fitx,ploty,undist_img,Minv)
       arr2[0,0].imshow(bin_threshold_img, 'gray')
       arr2[0,1].imshow(cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB))
       arr2[1,0].imshow(warpImage(bin_threshold_img,M),'gray')
       arr2[1,1].imshow(cv2.cvtColor(warped_img,cv2.COLOR_BGR2RGB))
       arr2[2,0].imshow(lane_plot_img)
       arr2[2,0].plot(left_fitx,ploty,color='blue')
       arr2[2,0].plot(right_fitx,ploty,color='blue')
       arr2[2,1].imshow(cv2.cvtColor(lane_drawn_img,cv2.COLOR_BGR2RGB)) 


   if 1:
       plt.show()

def getCameraPerspectiveMatrix():

   calibration_img_dir="/home/alok/Documents/udacity_nd/CarND-Advanced-Lane-Lines/camera_cal"
   mtx,dist = getCameraCalibrationMatrix(calibration_img_dir) 
   print('cal matrix:' )
   print(mtx)
   images = glob.glob("/home/alok/Documents/udacity_nd/CarND-Advanced-Lane-Lines/test_images/*.jpg")
    
   # perspective transform
   img_size=cv2.imread(images[0]).shape
   vertices_mask = np.float32([[150,img_size[0]-20],[580,450],[700,450],[1150,img_size[0]-20]],)
   dest_points = np.float32([[250,img_size[0]],[250,30],[1000,30],[1000,img_size[0]]])
   M = getPerspectiveTransform(vertices_mask,dest_points)
   Minv = getPerspectiveTransform(dest_points,vertices_mask) 

   return mtx,dist,M,Minv

def process_image(img,mtx,dist,M,Minv):
        
   undist_img = cv2.undistort(img,mtx,dist,None,mtx)
   # combined threshold image
   bin_threshold_img,color_bin_img = thresholdImage(undist_img)    
   warped_img = warpImage(undist_img,M)
   binary_warped_img = warpImage(bin_threshold_img,M)
   binary_histogram = binaryImgHistogram(binary_warped_img)
   left_lane_fit, right_lane_fit = slidingWindowSearch(binary_warped_img,binary_histogram)
   # Generate x and y values for plotting
   ploty = np.linspace(0, binary_warped_img.shape[0]-1, binary_warped_img.shape[0] )
   left_fitx = left_lane_fit[0]*ploty**2 + left_lane_fit[1]*ploty + left_lane_fit[2]
   right_fitx = right_lane_fit[0]*ploty**2 + right_lane_fit[1]*ploty + right_lane_fit[2]       
   # Generate image to plot lane lines on
   lcurve_m, rcurve_m = laneLineCurvature(left_fitx,right_fitx,ploty,(undist_img.shape[1],undist_img.shape[0]))
   img_with_lanes = drawOnPerspectiveImage(binary_warped_img,left_fitx,right_fitx,ploty,undist_img,Minv)

   return np.dstack((bin_threshold_img,bin_threshold_img,bin_threshold_img))*255#, lcurve_m, rcurve_m

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def transformVideo(clip,camera_mtx,camera_dist,M,Minv):
    def image_transform(image):
         return process_image(image,camera_mtx,camera_dist,M,Minv)
    return clip.fl_image(image_transform)


def processVideo(videoPath,outputDir):
    camera_mtx,camera_dist,M,Minv = getCameraPerspectiveMatrix()

    videoFileName = videoPath.split('/')[-1]
    print('video file name:',videoFileName)
    output = outputDir+'/out'+videoFileName
    print('out_video:',output)
    clip  = VideoFileClip(videoPath)
    processed_clip = clip.fx(transformVideo,camera_mtx,camera_dist,M,Minv)
    processed_clip.write_videofile(output,audio=False)


output_dir = "/home/alok/Documents/udacity_nd/CarND-Advanced-Lane-Lines/test_video_output"
video_list =["project_video.mp4","challenge_video.mp4","harder_challenge_video.mp4"]
video_path="/home/alok/Documents/udacity_nd/CarND-Advanced-Lane-Lines/"

#processVideo(video_path+video_list[1],output_dir )
main()
