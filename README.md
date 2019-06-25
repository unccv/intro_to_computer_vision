
![](graphics/computer_vision_cropped.png)


## Lectures
| Lecture |   Notebook/Slides | Required Reading/Viewing | Additional Reading/Viewing | Key Topics | 
| ------ | ------- | ------------- | --------------------------- | -------------------------- | 
| 1 | [A Brief History of Neural Networks](notebooks/A%20Brief%20History%20of%20Neural%20Networks.ipynb) | - | 
[Goodfellow Chapter 1](https://www.deeplearningbook.org/contents/intro.html) 
[fastai dk lesson 1](https://course.fast.ai/videos/?lesson=1)| 
Perceptrons, Multilayer Perceptrons, Neural Networks, The Rise of Deep Learning|
| Optional | [Introduction to Jupyter and Python](notebooks/Introduction%20to%20Jupyter%20and%20Python.ipynb) | - | 
[fastai ml lesson 1](http://course18.fast.ai/ml)| 
iPython, The Jupyter Notebook, Numpy, Matplotlib, Working with Image Data|
| 2 | [Computer Vision State of the Art](/welchlabs.io/unccv/intro_to_computer_vision/cv_applications.pptx) | [Alexnet Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) | 
- | 
State of the art in Classification, Detection, Pose Estimation, Image Generation, and other problems|
| 3 | [Computer Vision Applications](/welchlabs.io/unccv/intro_to_computer_vision/state_of_the_art_2019.pptx) | - | - | 
What can we do with comptuer vision?|


## Programming Challenges

### 1. Image Processing Script
In the challenge directory of this repo, you'll find a sample_student.py script. Your job is to complete the image processing methods in this script. 

[more instructions here, including autolab]

### 2. State-of-the-Art on Your Own Data
In this module, we'll quite a few State-of-the-Art computer vision algorithms. One of the really exciting things about computer vision right now is the amount of high quality, publically available code. For this part of your assignment, your job is to **run one publically avaialable algorithm on your own video or images**. Your deliverable is a short video, posted to YouTube, showing your results. For example, you could shoot your own video, and use and [Mask RCNN](https://github.com/matterport/Mask_RCNN) to process each frame, and stitch these results together into a short video. 

[Let's copy the table from the beginning of my Computer Vision Applications PPT here, we'll have to convert to markdown. We should add some submission instructions, also, would this be better as an individual or group assignment?]


## Setup
The Python 3 [Anaconda Distribution](https://www.anaconda.com/download) is the easiest way to get going with the notebooks and code presented here. 

(Optional) You may want to create a virtual environment for this repository: 

~~~
conda create -n cv python=3 
source activate cv
~~~

You'll need to install the jupyter notebook to run the notebooks:

~~~
conda install jupyter

# You may also want to install nb_conda (Enables some nice things like change virtual environments within the notebook)
conda install nb_conda
~~~

This repository requires the installation of a few extra packages, you can install them with:

~~~
conda install -c pytorch -c fastai fastai
conda install jupyter
conda install opencv
pip install jupyterthemes
~~~

(Optional) [jupyterthemes](https://github.com/dunovank/jupyter-themes) can be nice when presenting notebooks, as it offers some cleaner visual themes than the stock notebook, and makes it easy to adjust the default font size for code, markdown, etc. You can install with pip: 

~~~
pip install jupyterthemes
~~~

Recommend jupyter them for **presenting** these notebook (type into terminal before launching notebook):
~~~
jt -t grade3 -cellw=90% -fs=20 -tfs=20 -ofs=20 -dfs=20
~~~

Recommend jupyter them for **viewing** these notebook (type into terminal before launching notebook):
~~~
jt -t grade3 -cellw=90% -fs=14 -tfs=14 -ofs=14 -dfs=14
~~~



