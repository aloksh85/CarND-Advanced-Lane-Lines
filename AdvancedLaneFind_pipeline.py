
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
import os
from common_parameters import XM_PER_PIX,YM_PER_PIX
from Line import Line

def getCameraCalibrationMatrix(distortedImgDir, nx=9, ny=6 , redo_calibration=False):
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
    mtx =  None
    dist = None
    rvecs = None
    tvecs = None

    if not os.path.isfile("/home/alok/Documents/udacity_nd/CarND-Advanced-Lane-Lines/camera_calibration.p") or redo_calibration:
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
        
        if ret_mtx:
            print('Writing camera calibration to file')
            camera_dict = {'cam_cal_mtx':mtx,'distortion':dist}
            pickle.dump(camera_dict,open("camera_calibration.p","wb"))
    else:
        print('Loading camera calibration from file')
        camera_dict = pickle.load(open("camera_calibration.p","rb"))
        mtx =  camera_dict['cam_cal_mtx']
        dist = camera_dict['distortion']  

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

def thresholdImage(img,sx_thresh=(20,100),s_thresh=(110,130),l_thresh=(200,250),b_thresh=(95,110)):
    #print('r: ',np.min(img[:,:,2]),'-',np.max(img[:,:,2]),
    #'\ng:',np.min(img[:,:,1]),'-',np.max(img[:,:,1]),
    #'\nb:',np.min(img[:,:,0]),'-',np.max(img[:,:,0]))
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    #print('h: ',np.min(img[:,:,0]),'-',np.max(img[:,:,0]),
    #'\nl:',np.min(img[:,:,1]),'-',np.max(img[:,:,1]),
    #'\ns:',np.min(img[:,:,2]),'-',np.max(img[:,:,2]))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:,:,2]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    #print('l:',np.min(lab[:,:,0]),'-',np.max(img[:,:,0]),
    #        '\na:',np.min(img[:,:,1]),'-',np.max(img[:,:,1]),
    #        '\nb:',np.min(img[:,:,2]),'-',np.max(img[:,:,2]))
    
    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(gray) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel > l_thresh[0]) & (l_channel <=l_thresh[1])] = 1

    b_binary =np.zeros_like(b_channel)
    b_binary[(b_channel > b_thresh[0]) & (b_channel <= b_thresh[1])] = 1
    
    lsx_combined = np.zeros_like(sxbinary)
    lsx_combined = l_binary+sxbinary
    lsx_binary = np.zeros_like(lsx_combined)
    lsx_binary[lsx_combined >1]=1
    #print('lsx_combined:',np.min(lsx_combined),'-',np.max(lsx_combined))
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((sxbinary , (l_binary), (b_binary))) * 255
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(b_binary ==1)| (l_binary==1)] = 1

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


def slidingWindowSearch(binary_warped,bin_hist,left_lane,right_lane,nwindows=9,win_margin=100, recenter_minpix = 50):

    #out_img = np.copy(binary_warped)
    out_img = np.dstack((binary_warped,binary_warped,binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(bin_hist.shape[0]/2)
    leftx_base = np.argmax(bin_hist[0:midpoint])
    rightx_base = np.argmax(bin_hist[midpoint:]) + midpoint
   
    if not left_lane.peakx:
        left_lane.peakx = [leftx_base]
    elif len(left_lane.peakx) < 3:
        left_lane.peakx.append(leftx_base)
    else:
        mean_leftpeak = np.int(np.mean(left_lane.peakx[-3]))
        if np.abs(leftx_base - mean_leftpeak) > 3:
            leftx_base=left_lane.peakx[-1]
        else:
            left_lane.peakx.append(leftx_base)


    
    if not right_lane.peakx:
        right_lane.peakx = [rightx_base]
    elif len(right_lane.peakx) < 3:
        right_lane.peakx.append(rightx_base)
    else:
        mean_rightpeak = np.mean(right_lane.peakx[-3:])
        if np.abs(rightx_base - mean_rightpeak) > 3:
            rightx_base = right_lane.peakx[-1]
        else:
            right_lane.peakx.append(rightx_base) 


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
    
    left_empty_windows = 0
    right_empty_windows = 0
    deltax_left = 0.0
    deltax_right = 0.0
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
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        
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
            leftx_new = np.int(np.mean(nonzerox[good_left_inds]))
            deltax_left = np.int(leftx_new - leftx_current)
            leftx_current = leftx_new
            left_empty_windows = 0

            
        if len(good_right_inds) > minpix:        
             rightx_new= np.int(np.mean(nonzerox[good_right_inds]))
             deltax_right = np.int(rightx_new - rightx_current)
             rightx_current = rightx_new
             right_empty_windows = 0

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
    right_fit= np.polyfit(righty, rightx, 2)
  
    
    return left_fit, right_fit,(leftx,lefty), (rightx,righty),out_img


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
    lane_center_px = int(np.mean(x_right[-3:] - x_left[-3:]))
    offset_px = int(img_size[0]*0.5) - lane_center_px
    offset_m = offset_px*x_m_px


    return offset_m

def weighted_img(img, initial_img, α=0.5, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
def main():

   calibration_img_dir="/home/alok/Documents/udacity_nd/CarND-Advanced-Lane-Lines/camera_cal"

   mtx,dist = getCameraCalibrationMatrix(calibration_img_dir) 
   print('cal matrix:' )
   print(mtx)
   images = glob.glob("/home/alok/Documents/udacity_nd/CarND-Advanced-Lane-Lines/test_images/*.jpg")
   left_lane =Line()
   right_lane = Line()
   # perspective transform
   img_size=cv2.imread(images[0]).shape
   vertices_mask = np.float32([[150,img_size[0]-30],[580,450],[700,450],[1150,img_size[0]-30]])
   dest_points = np.float32([[300,img_size[0]],[300,30],[1150,30],[1150,img_size[0]]])
   
   print('src points:\n',vertices_mask)
   print('dst points:\n',dest_points)

   M = getPerspectiveTransform(vertices_mask,dest_points)
   Minv = getPerspectiveTransform(dest_points,vertices_mask) 

   
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
       cv2.line(undist_img, (vertices_mask[0,0],vertices_mask[0,1]),
         (vertices_mask[1,0],vertices_mask[1,1]), [0,0,255], 2)
       cv2.line(undist_img, (vertices_mask[1,0],vertices_mask[1,1]),
         (vertices_mask[2,0],vertices_mask[2,1]), [0,0,255], 2)
       cv2.line(undist_img, (vertices_mask[2,0],vertices_mask[2,1]),
         (vertices_mask[3,0],vertices_mask[3,1]), [0,0,255], 2)
       cv2.line(undist_img, (vertices_mask[3,0],vertices_mask[3,1]),
         (vertices_mask[0,0],vertices_mask[0,1]), [0,0,255], 2)       
      
       warped_img = warpImage(undist_img,M)
       binary_warped_img = warpImage(bin_threshold_img,M)
       binary_histogram = binaryImgHistogram(binary_warped_img)
       left_lane_fit, right_lane_fit,left_inds,right_inds,window_overlay_img = slidingWindowSearch(binary_warped_img,binary_histogram,left_lane,right_lane)
       
       # Generate x and y values for plotting
       ploty = np.linspace(0, binary_warped_img.shape[0]-1, binary_warped_img.shape[0] )
       left_fitx = left_lane_fit[0]*ploty**2 + left_lane_fit[1]*ploty + left_lane_fit[2]
       right_fitx = right_lane_fit[0]*ploty**2 + right_lane_fit[1]*ploty + right_lane_fit[2]       
       # Generate image to plot lane lines on
       lcurve, rcurve = laneLineCurvature(left_fitx,right_fitx,ploty,(undist_img.shape[1],undist_img.shape[0]),XM_PER_PIX,YM_PER_PIX)
       offset_m = get_offset(ploty,left_fitx,right_fitx,XM_PER_PIX,YM_PER_PIX,(undist_img.shape[1],undist_img.shape[0]))
       print('left lane curvature: ',lcurve)
       print('right lane curvature: ',rcurve)
       print('offset_m: ',offset_m)
       lane_plot_img = np.dstack((binary_warped_img,binary_warped_img,binary_warped_img))*255       
       lane_drawn_img = drawOnPerspectiveImage(binary_warped_img,left_fitx,right_fitx,ploty,undist_img,Minv)
       arr2[0,0].imshow(bin_threshold_img, 'gray')
       arr2[0,1].imshow(cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB))
       arr2[1,0].imshow(color_bin_img)
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

   vertices_mask = np.float32([[250,img_size[0]-50],[580,450],[700,450],[1050,img_size[0]-50]])
   dest_points = np.float32([[400,img_size[0]],[400,0],[img_size[1]-400,0],[img_size[1]-400,img_size[0]]])
   M = getPerspectiveTransform(vertices_mask,dest_points)
   Minv = getPerspectiveTransform(dest_points,vertices_mask) 

   return mtx,dist,M,Minv

def localLaneSearch(binary_warped_img, left_lane,right_lane,margin =50 ):

    out_img =np.dstack((binary_warped_img,binary_warped_img,binary_warped_img))
    nonzero = binary_warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_fit  = left_lane.current_fit
    right_fit = right_lane.current_fit

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

# Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
# Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

# update left and right lane 
    #lpolyfit_diff_mag = np.linalg.norm(left_lane.best_fit - left_fit)
    #rpolyfit_diff_mag = np.linalg.norm(right_lane.best_fit - right_fit)
    
    #if lpolyfit_diff_mag > 5.0:
    #    left_fit = left_lane.best_fit
    #else:
    #    left_lane.current_fit = left_fit
    #    left_lane.poly_fits =np.vstack((left_lane.poly_fits,left_fit))
    #    left_lane.best_fit = np.mean(left_lane.poly_fits[-3:,:],axis=0)

    #if rpolyfit_diff_mag > 5.0:
    #    right_fit = right_lane.best_fit
    #else:
    #    right_lane.current_fit = right_fit
    #    right_lane.poly_fits = np.vstack((right_lane.poly_fits,right_fit))
    #    right_lane.best_fit = np.mean(right_lane.poly_fits[-3:,:],axis=0)

    left_lane.current_fit = left_fit
    right_lane.current_fit = right_fit
# Generate x and y values for plotting
    
    ploty = np.linspace(0, binary_warped_img.shape[0]-1, binary_warped_img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
				  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
# Draw the lane onto the warped blank image
    window_img = np.zeros_like(out_img)

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    
    result = cv2.addWeighted(out_img, 1, window_img, 0.2, 0)
    return ploty, left_fitx, right_fitx, (leftx,lefty), (rightx,righty),result

def process_image(img,mtx,dist,M,Minv,left_lane,right_lane):
       undist_img = cv2.undistort(img,mtx,dist,None,mtx)
       # combined threshold image
       bin_threshold_img,color_bin_img = thresholdImage(undist_img)    
       warped_img = warpImage(undist_img,M)
       binary_warped_img = warpImage(bin_threshold_img,M)
       binary_histogram = binaryImgHistogram(binary_warped_img)
     
       ploty,left_fitx,right_fitx,left_inds,right_inds = None,None,None,None,None
       
       if not (left_lane.detected and right_lane.detected):
           left_lane_fit, right_lane_fit, left_inds, right_inds,window_overlay_img = slidingWindowSearch(binary_warped_img,binary_histogram,left_lane,right_lane,win_margin=90)
       # Generate x and y values for plotting
           ploty = np.linspace(0, binary_warped_img.shape[0]-1, binary_warped_img.shape[0] )
           left_fitx = left_lane_fit[0]*ploty**2 + left_lane_fit[1]*ploty + left_lane_fit[2]
           right_fitx = right_lane_fit[0]*ploty**2 + right_lane_fit[1]*ploty + right_lane_fit[2]                 
           left_lane.recent_xfitted = left_fitx
           right_lane.recent_xfitted = right_fitx
           left_lane.poly_fits=np.vstack((left_lane.poly_fits,left_lane_fit))
           right_lane.poly_fits=np.vstack((right_lane.poly_fits,right_lane_fit))
           left_lane.current_fit = left_lane_fit
           right_lane.current_fit = right_lane_fit

           if left_lane.poly_fits.shape[0] >=3 and right_lane.poly_fits.shape[0] >=3:
               left_lane.detected = True
               right_lane.detected = True
               left_lane.best_fit = np.mean(left_lane.poly_fits[-3:,:],axis=0)
               right_lane.best_fit = np.mean(right_lane.poly_fits[-3:,:],axis=0)
               #print('Full slide search')

       else:
           #print('local lane search')
           ploty,left_fitx,right_fitx,left_inds,right_inds,window_overlay_img = localLaneSearch(binary_warped_img,left_lane,right_lane)
           

       # Generate image to plot lane lines on
       lcurve_m, rcurve_m = laneLineCurvature(left_fitx,right_fitx,ploty,(undist_img.shape[1],undist_img.shape[0]),XM_PER_PIX,YM_PER_PIX)
       img_with_lanes = drawOnPerspectiveImage(binary_warped_img,left_fitx,right_fitx,ploty,undist_img,Minv)
       bin_warped_img_3ch = np.dstack((binary_warped_img,binary_warped_img,binary_warped_img))*255
       bin_thresh_img_3ch = np.dstack((bin_threshold_img,bin_threshold_img,bin_threshold_img))*255
       wt_img = weighted_img(undist_img,color_bin_img)
       l_image_3ch = np.dstack((color_bin_img[:,:,1],color_bin_img[:,:,1],color_bin_img[:,:,1]))
       s_image_3ch =np.dstack((color_bin_img[:,:,2],color_bin_img[:,:,2],color_bin_img[:,:,2]))
       b_image_3ch =np.dstack((color_bin_img[:,:,0],color_bin_img[:,:,0],color_bin_img[:,:,0]))

       result_img = np.vstack((np.hstack((undist_img,color_bin_img)),
                               np.hstack((bin_warped_img_3ch,img_with_lanes))))
       return img_with_lanes#result_img #out_img 

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def transformVideo(clip,camera_mtx,camera_dist,M,Minv,left_lane,right_lane):
    def image_transform(image):
         return process_image(image,camera_mtx,camera_dist,M,Minv,left_lane,right_lane)
    return clip.fl_image(image_transform)


def processVideo(videoPath,outputDir):
    camera_mtx,camera_dist,M,Minv = getCameraPerspectiveMatrix()
    left_lane  = Line()
    right_lane = Line()
    print('M matrix: \n',M) 
    videoFileName = videoPath.split('/')[-1]
    print('video file name:',videoFileName)
    output = outputDir+'/out'+videoFileName
    print('out_video:',output)
    clip  = VideoFileClip(videoPath)#.subclip(0,10)
    processed_clip = clip.fx(transformVideo,camera_mtx,camera_dist,M,Minv,left_lane,right_lane)
    processed_clip.write_videofile(output,audio=False)


output_dir = "/home/alok/Documents/udacity_nd/CarND-Advanced-Lane-Lines/test_video_output"
video_list =["project_video.mp4","challenge_video.mp4","harder_challenge_video.mp4"]
video_path="/home/alok/Documents/udacity_nd/CarND-Advanced-Lane-Lines/"

processVideo(video_path+video_list[0],output_dir )
#processVideo(video_path+video_list[1],output_dir
main()
