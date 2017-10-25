## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goal of the project was to write a software pipeline to identify the lane boundaries in a video. Here I will elaborate how I implemented the pipeline. I will address all requirements in the project rubric below.


### Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
To compute the calibration matrix, I used Open CV  methods `findChessboardCorners`, `calibrateCamera` and `undistort`. Since this calibration is required to be done only once for a camera and lense, I saved the calibration matrix and distortion measure to a pickle format file. While developing the pipeline, I load the saved matrix and use it for undistorting all images. A sample of undistorted checkerboard images are available below:
![Undistorted Images](undistorted_img.png)

### Pipeline
The different stages of image processing are explained below 
####Thresholding 
In order to obtain a binary image containing likely image pixels, I explored the following
* Computing grayscale image derivative in x-direction using the *Sobel*  operator
* Converting to HLS color space and using thresholds on the **S-channel** and **L-channel**
* Converting to LAB color space and ysing threshold on the **B-channel**

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

