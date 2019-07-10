## ------------------------- ##
##
## evaluate.py
## Basic image processing utilties.
## 
##
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
img = cv2.imread('messi.jpg')
img_pil = Image.open("messi.jpg")



print("Checking Gray Scale...")
##### Gray Scale ##########

# Convert
eval_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Get Student ans
stu_img = ss.convert_to_grayscale(img)

# Validate images
if(len(stu_img.shape)==2):
    if(eval_img.all()==stu_img.all()):
        score += 1
else:
    score += 0
print("Score: =", score)



print("Checking crop Image...")
##### Crop Image ##########
crop_area = (40, 50, 300, 400)
# Convert
eval_img = img_pil.crop((crop_area[0],crop_area[1],crop_area[1]+crop_area[2],crop_area[0]+crop_area[3]))

# Get Student ans
stu_img = ss.crop_image(img, crop_area)

# Validate images
if (np.array(eval_img)).all() == stu_img.all():
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

eval_contrast_image = cv2.imread("maximum_contrast_image.png")
# Validate images

if stu_maximum_contrast.all() == eval_contrast_image.all():
    score +=1
else:
    score +=0
print("Score: =", score)



print("Checking Flip Image...")
##### Flipped Image Image ##########
eval_horizontal_img = cv2.flip( img, 0 )
eval_vertical_img = cv2.flip( img, 1 )

direction = "horizontal"

# Get Student ans
stu_flip_img = ss.flip_image(img, direction)

# Validate images
if direction=='vertical':
    if(stu_flip_img.shape == eval_vertical_img.shape):
        if(stu_flip_img.all() == eval_vertical_img.all()):
            score +=1
elif direction=='horizontal':
    if(stu_flip_img.shape == eval_horizontal_img.shape):
        if(stu_flip_img.all() == eval_horizontal_img.all()):
            score +=1
else:
    score +=0
print("Score: =", score)



print("Checking Pixels above Threshold...")
##### Pixels above threshold ##########

#threshold value and pixel count
threshold = 250
pixel_count = 7558

eval_horizontal_img = cv2.flip( img, 0 )
eval_vertical_img = cv2.flip( img, 1 )

# Get Student ans
stu_pixel_count = ss.count_pixels_above_threshold(img, threshold)

if stu_pixel_count == pixel_count:
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
    if(stu_norm_image.all()==eval_norm_image.all()):
        score +=1
    else:
        score +=0
else:
    score +=0
print("Score: =", score)



print("Checking for Resized image...")
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
    if(eval_resized.all()== stu_resize_image.all()):
            score += 1
    else:
            score += 0
else:
    score +=0
print("Score: =", score)



program_end = time.time()
complete_time = program_end - program_start
omplete_time = round(complete_time, 5)
print("Program completetion time (seconds): = ", complete_time)
print("Total Score: =", score,"/7")
