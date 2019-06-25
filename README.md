
![](graphics/computer_vision_cropped.png)

## Introduction to Computer Vision



1. Introduction to Computer Vision 
2. The State of the Art, Applications, and Using Open Source Code - let's use slides for this.
3. (Optional/Review Session) Introduction to Jupyter + Python of Compter Vision 
4. How to do things with fastai


## Lectures

|   Notebook/Slides | Recommended Reading/Viewing | Additional Reading/Viewing | Key Topics | 
| ------- | ------------- | --------------------------- | -------------------------- | 
[A Brief History of Neural Networks](http://www.welchlabs.io/unccv/the_original_problem/notebooks/top.html) | [The Summer Vision Project](papers/summer_vision_project.pdf) | - | - |


## Programming Challenge


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



