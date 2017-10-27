## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goal of the project was to write a software pipeline to identify the lane boundaries in a video. Here I will elaborate how I implemented the pipeline. I will address all requirements in the project rubric below.

Note: The pipeline has been implemented in file called **AdvancedLaneFind _pipeline.py**
### Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
To compute the calibration matrix, I used Open CV  methods `findChessboardCorners`, `calibrateCamera` and `undistort`. Since this calibration is required to be done only once for a camera and lense, I saved the calibration matrix and distortion measure to a pickle format file. While developing the pipeline, I load the saved matrix and use it for undistorting all images. A sample of undistorted checkerboard images are available below:
![Undistorted Images](undistorted_image.png)

### Pipeline
The different stages of image processing are explained below 
####Thresholding 
In order to obtain a binary image containing likely image pixels, I explored the following
* Computing grayscale image derivative in x-direction using the **Sobel**  operator
* Converting from RGB to HLS color space and using thresholds on the **S-channel** and **L-channel**
* Converting RGB to LAB color space and using threshold on the **B-channel**

The sobel operator responds well to intensity changes so it is able to capture the lane lines but also activates when there are shadows and other illumination changes. 

The S-channel from HSV space and B-channel from LAB space capture the yellow lane line very well. However S-channel activates in low lighting conditions too whereas the B-channel consistently captures the yellow lines with low noise. 

The L-channel from HLS space captures the white lane lines well but also responds to changes in illumination in the image.

Some thresholding results are shown below
![Thresholding Result](output_images/binary_test2.jpg)
After observing the thresholding results I decided to use a combination of thresholding on L-channel and B-Channel to identify yellow and white lane lines. The method `thresholdImage` returns the thresholded binary image

###Perspective Transform
To transform the camera images from perspective into bird-eye view I used OpenCV method `warpPerspective`. To calculate the perspective transform and inverse-perspective transform I chose 4 points  from one of the test images having straight lane lines and specified 4 corresponding destination points. The source and destination points are hard-coded as :

	
    src_pts = np.float32([[250,img_height-50],
                                 [580,450],
                                 [700,450],
                                 [1050,img_height-50]])
    
    dest_points = np.float32([[400,img_height],
                             [400,0],
                             [img_width-400,0],
                             [img_width-400,img_height]])

A sample of warped images using the perspective transform is available below
![Warped Images](output_images/warp_straight_lines1.jpg)

### Lane pixel identification
To identify lan	e line pixels in warped binary image, I sum number of non-zero (white) pixels in each column of lower half of image. The columns with maximum relevant pixels on the left half and  right half of image are recorded as the base of lane line.
Once the base location of left and right lane line has been identified, a sliding window search is used to identify lane line pixels. The number of windows to be used, and window width are configurable. The sliding search is implemented in method called `slidingWindowSearch`

The search algorithm works is explained below:
1. Start at base positions of left and right line - create window using width and height
2. Finds all pixels lying inside the current window and add them to a list of points that form the lane line
3. If number of pixels exceed a set threshold, the x_position of next window center is updated to centroid of pixels found
4. Update window location, goto step 2 and repeat until till entire height of image is covered

After the search,  a second degree polynomial is fit to coordinates of the pixels that belong to lane lines. Polynomial fitting is achieved using numpy method `polyfit` which returns co-efficients (A,B &C) of the second degree polynomial - `f(y) = Ay^2 +By+C`

 ![Slide_search](output_images/search_test4.jpg)
### Calculation of radius of curvature and offset
 The radius of curvature is calculated as : `R = ((1+(2Ay+B)^2))^1.5)/|2A|` 
 The method `lanelLineCurvature` computes radius of curvature. It uses y=image_height to compute the radius at base of the image
 
 Offset of vehicle from lane center - It is assumed that camera is mounted at center of car such that the lane center is the midpoint at the bottom of the image between the two detected lane lines. Hence, the offset of lane center from  image center is the vehicle offset from lane center. The method `get_offset` computes the vehicle offset.

To transform the radius and offset from pixel space to metric units, I use the following mappings 
 _YMetersperpixel_= 30/720
 _XMetersperpixel_=3.7/700
 
 These mappings are based on US regulations that require a minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each.
 
###Lane finding results on Test Images
 The method `main` runs all images in the test_images folder through the pipeline explained above. Results are shown below. The pipeline is able to detect the lane lines fairly well. All processed test images are also available in the _output\_images_ directory
 ![test image results](output_images/test_image_results.png)
 
###  Processing video using the pipeline
 The result of running the pipeline on the test video are available in the directory [test video directory](/test_video_output) 
 
 

