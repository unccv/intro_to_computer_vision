## ------------------------- ##
##
## evaluate.py
## Basic image processing utilties.
##
##In order to test your code, you can use the images in the "data" folder. But for the actual evaulation, we will use different images. The variables
## 'direction', 'threshold', 'pixel_count' will have different values. Any hard coded value here is only for you to test sample_student.py on your local machine.
##
##Here we check for 99% match between the images returned from sample_student.py and our evaluation images.
##
##In order to run this evaluation script locally, you have will have to install opencv(cv2) and pillow(PIL) package.
##This can be done installed using pip by simply typing : "pip install pillow" and "pip install opencv-python" on your terminal.
## ------------------------- ##

import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from PIL import Image


#Import student's methods:
import sample_student as ss

program_start = time.time()


###SCORE########
score = 0

#### Read Image ######
img = cv2.imread('../data/messi.jpg')
img_pil = Image.open("../data/messi.jpg")


print("Checking Gray Scale...")
##### Gray Scale ##########

# Convert
eval_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


stu_img = ss.convert_to_grayscale(img)

# Validate images
if(len(stu_img.shape)==2):
    if(np.sum(np.abs(stu_img))/np.sum(eval_img) >= 0.90):
        score += 1
else:
    score += 0
print("Score: =", score)



print("Checking crop Image...")
##### Crop Image ##########
crop_area = (40, 50, 300, 400)
# Convert
eval_crop_img = img_pil.crop((crop_area[0],crop_area[1],crop_area[1]+crop_area[2],crop_area[0]+crop_area[3]))

# Get Student ans
stu_crop_img = ss.crop_image(img, crop_area)

# Validate images
if (np.sum(np.array(eval_crop_img)/np.sum(np.abs(stu_crop_img))) >= 0.99):
    score +=1
else:
    score +=0
print("Score: =", score)



print("Checking Contrast Range of Image...")
##### Contrast Range Image ##########
eval_contrast_range = img_pil.getextrema()
diff = []
diff.append(abs(eval_contrast_range[0][0]-eval_contrast_range[0][1]))
diff.append(abs(eval_contrast_range[1][0]-eval_contrast_range[1][1]))
diff.append(abs(eval_contrast_range[2][0]-eval_contrast_range[2][1]))

# Get Student ans
stu_contrast_range = ss.compute_range(img)

# Validate images
if diff == list(stu_contrast_range):
    score +=1
else:
    score +=0
print("Score: =", score)



print("Checking for Maximum Contrast...")
##### Maximum Contrast ##########

# Get Student ans
stu_maximum_contrast = ss.maximize_contrast(img,[0,255])

eval_contrast_image = cv2.imread("../data/maximum_contrast_image.png")
# Validate images

if(np.sum(eval_contrast_image)/ np.sum(np.abs(stu_maximum_contrast)) >= 0.99):
    score +=1
else:
    score +=0
print("Score: =", score)



print("Checking Flip Image...")
##### Flipped Image Image ##########
eval_horizontal_img = cv2.flip( img, 1 )
eval_vertical_img = cv2.flip( img, 0 )

direction = "vertical"

# Get Student ans
stu_flip_img = ss.flip_image(img, direction)

# Validate images
if direction=='vertical':
    if(stu_flip_img.shape == eval_vertical_img.shape):
        if(np.array_equal(eval_vertical_img, stu_flip_img)):
            score +=1
elif direction=='horizontal':
    if(stu_flip_img.shape == eval_horizontal_img.shape):
        if(np.array_equal(eval_horizontal_img, stu_flip_img)):
            score +=1
else:
    score +=0
print("Score: =", score)



print("Checking Pixels above Threshold...")
##### Pixels above threshold ##########

#threshold value and pixel count
threshold = 250
pixel_count = 4543

# Get Student ans
stu_pixel_count = ss.count_pixels_above_threshold(img, threshold)
if (stu_pixel_count <= (pixel_count+2000) and stu_pixel_count >= (pixel_count-2000)):
    score +=1
else:
    score +=0
print("Score: =", score)



print("Checking for normalized image...")
##### Normalized Image ##########

eval_norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Get Student ans
stu_norm_image = ss.normalize(img)

if(stu_norm_image.shape == eval_norm_image.shape):
    if(np.sum(eval_norm_image)/ np.sum(np.abs(stu_norm_image)) >= 0.99):
        score +=1
    else:
        score +=0
else:
    score +=0
print("Score: =", score)




print("Checking for Resized image... (BONUS PROBLEM)")
##### Resiszed Image ##########

#scale factor
scale_factor = 2

# Get Student ans
stu_resized_image = ss.resize_image(img, scale_factor)

#scale processing
scale_percent = scale_factor * 100 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
eval_resized_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

if(eval_resized_image.shape == stu_resized_image.shape):
    if(np.sum(np.abs(stu_resized_image))/np.sum(eval_resized_image) > 0.99):
            score += 1
    else:
            score += 0
else:
    score +=0
print("Score: =", score)



program_end = time.time()
complete_time = program_end - program_start
complete_time = round(complete_time, 5)
print("Program completetion time (seconds): = ", complete_time)
print("Total Score: =", score,"/7")
