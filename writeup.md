
## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---
### Writeup / README

#### [writeup.md](https://github.com/veegit/CarND-Vehicle-Detection/blob/master/writeup.md) 


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Code for extracting HOG_features which was pretty much taken from the project videos

````
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features
````
This uses the `skimage.feature.hog` method from skimage library

This method was called from `extract_features` method which fetches the HOG features for all channels on the images from training data. The following image below shows the HOG features for YUV channels. The reason for choosing YUV channel is explained later

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text](https://raw.githubusercontent.com/veegit/CarND-Vehicle-Detection/master/report_images/yuv_hog.png)

#### 2. Explain how you settled on your final choice of HOG parameters.

I used the exact HOG parameters mentioned in the project video and it gave excellent results on the test images and test video. Only thing I modified was color space, I tried RGB, HLS and YUV space. YUV gave me better result on test images

#### RGB
![alt text](https://raw.githubusercontent.com/veegit/CarND-Vehicle-Detection/master/report_images/rgb.png)

#### YUV
![alt text](https://raw.githubusercontent.com/veegit/CarND-Vehicle-Detection/master/report_images/yuv.png)


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used the same linear classifer provided in project videos code and used all the color histogram, bin spatial and HOG 3-channel YUV features. 

````
svc = LinearSVC()
````

````
27.72 Seconds to train SVC...
Test Accuracy of SVC =  0.9892
My SVC predicts:  [ 0.  0.  0.  0.  0.  1.  1.  0.  0.  0.]
For these 10 labels:  [ 0.  0.  0.  0.  0.  1.  1.  0.  0.  0.]
0.00681 Seconds to predict 10 labels with SVC
````
##### Example of car vs non-car feature

![alt text](https://raw.githubusercontent.com/veegit/CarND-Vehicle-Detection/master/report_images/car-vs-noncar.png)

##### These are  the bin spatial features for RGB vs YUV 

![alt text](https://raw.githubusercontent.com/veegit/CarND-Vehicle-Detection/master/report_images/bin-spatial-rgbvsyuv.png)

##### Also how normalization of data changed the distribution

![alt text](https://raw.githubusercontent.com/veegit/CarND-Vehicle-Detection/master/report_images/raw-vs-normalized.png)

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding windows search was implemented in the `find_cars` method. 

| # | Step |
| ------------- | ------------- |
| 1 | Find Image Area to investigate (whi as abovech is from ytop = 400 to ybottom = 656) |
| 2 | Define blocks and steps with 64 as the orginal sampling rate, with 8 cells and 8 pix per cell |
| 3 | Compute hog features for each channel on whole image |
| 4 | Extract the hog features, color features for the region defined in 1 |
| 5 | Transform the data | 
| 6 | Run the classifier on the transformed data and get prediction |
| 7 | If prediction is a car, the get the rectangle coordinates and add it to the list |
| 8 | Send the rectangles to draw the bounding boxes as per `boxed_image` method |
| 9 | Run the aboves steps for all scales |
| 10 | Generate heatmap for all boxes `add_heat` |
| 11 | Apply threshold to help remove false positives on heatmap |
| 12 | Find final boxes from heatmap using label function and draw final box `draw_labeled_bboxes` |

This is an example of different scales drew the boxes around the image
![alt text](https://raw.githubusercontent.com/veegit/CarND-Vehicle-Detection/master/report_images/scales.png)

This is image after step 8

![alt text](https://raw.githubusercontent.com/veegit/CarND-Vehicle-Detection/master/report_images/sliding-window.png)

This is image after step 12

![alt text](https://raw.githubusercontent.com/veegit/CarND-Vehicle-Detection/master/report_images/pipleline.png)




#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used five scales from `scales = [1,1.33, 1.5, 2, 3]` and threshold for heatmap as 4. The following was ran on test images


![alt text](https://raw.githubusercontent.com/veegit/CarND-Vehicle-Detection/master/report_images/yuv.png)
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I created a `Car` class to store recent heatmaps over last `max_frames = 10` frames. Last 10 frames would be used to create an average heatmap which we will threshold and create labelled boxes around it

This is the code which does that

````
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat,boxes)

    heat = np.expand_dims(heat, axis=0)
    if (car.recent_heat.shape[0] >= max_frames):
        car.recent_heat = np.delete(car.recent_heat, 0, 0)
        
    car.recent_heat = np.append(car.recent_heat,heat,axis=0)
    avg_heat = np.average(car.recent_heat, axis=0)
    # Apply threshold to help remove false positives
    heat = apply_threshold(avg_heat,threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

````

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Pipeline is very sensitive to threshold over heatmap, it's very difficult to come up with a good threshold to have a fine balance in reducing false positives and false negatives. Moreover the speed of the car had an impact on sliding window, a fast moving car will be in less frames and will only be detected few times. Also when neighbouring two cars were driving together it created a shape which was not like a car and confounded the classfier for a brief time.

