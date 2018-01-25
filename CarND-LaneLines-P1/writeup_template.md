# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I apply Gaussian blur to the gray-scale images. The 3rd step is to apply Canny edge identification algorithm to the image with low threshold as 50 and high threshold as 120. Next the image of edges will be cropped with the region of interest as a polygon, of which the vertices are (0.075 of width, height), (0.42 of width, 0.625 of height), (0.58 of width, 0.625 of height) and (0.95 of width, height). The last step is to apply Hough on edge detected image to retrieve line segments and combine it with the original image.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by first dividing all end points of detected line segments into left lane and right lane. It can be achieved by checking the slope of the line segments. Since the original point is at the top left corner, positive slope corresponds to right land and negative to left lanes. Furthermore all line segments with absolute value of slope greater than 2 or less than 0.3 will be considered as "not a valide lane" and will be ignored. 2 separate lists are created to collect points of left and right lanes. After going through end points of all detected line segments, I apply linear fit to both lists and acquire the formula of lines both left and right lane. Finally by calculating the 4 intercept points between left and right lane with the boundary of region of interest, the function is able to draw the singal left and right lanes.


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the road condition is not good and images from camera are not quite stable. 

Another shortcoming could be that this method can not work when vehicle enters corners or either of the lanes are not straight. 


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to apply polynomial fits of higher degrees of freedom so it can fit curves of single shape.

Another possible improvement would be apply a new method of way to draw lines such as connecting line segments instead of applying linear fit.
