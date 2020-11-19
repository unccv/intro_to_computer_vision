
![](graphics/computer_vision_cropped.png)




### 1. State-of-the-Art on Your Own Data
In this module, we'll quite a few State-of-the-Art computer vision algorithms. One of the really exciting things about computer vision right now is the amount of high quality, publically available code. For this part of your assignment, your job is to **run one publically avaialable algorithm on your own video or images**. Your deliverable is a short video, posted to YouTube, showing your results. For example, you could shoot your own video, and use and [Mask RCNN](https://github.com/matterport/Mask_RCNN) to process each frame, and stitch these results together into a short video. 

   #### A Sample of The Computer Vision State of the Art in 2019

| PROBLEM | PAPER | CODE |
| :---:         |     :---:      |          :---: |
| Classification| [“ResNet” Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)| Implemented in keras, pytorch, fastai |
| Detection     |[RetinaNet: Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)<br><br><br>[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)<br><br><br> [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)<br><br><br>[YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)|Part of FAIR’s [Detectron](https://github.com/facebookresearch/Detectron)<br><br><br>Part of [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)<br><br><br>Part of [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)<br><br><br>[CODE](https://pjreddie.com/darknet/yolo/)|
|Semantic Segmentation| [“Deeplab v3” Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1708.02002.pdf)|[CODE](https://github.com/tensorflow/models/tree/master/research/deeplab)|
|Instance Segmentation| [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)|[CODE](https://github.com/matterport/Mask_RCNN)|
|Human Pose Estimation| [OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/pdf/1812.08008.pdf)|[CODE](https://github.com/CMU-Perceptual-Computing-Lab/openpose)|
|Hand Pose Estimation| [GANerated Hands for Real-Time 3D Hand Tracking from Monocular RGB](https://arxiv.org/pdf/1712.01057.pdf)| |
|Face Detection| [Selective Refinement Network for High Performance Face Detection](https://arxiv.org/pdf/1809.02693v1.pdf)|[CODE](https://github.com/ChiCheng123/SRN)|
|Face Recognition| [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832v3.pdf)|[CODE](https://github.com/davidsandberg/facenet)|
|Tracking| [Fast Online Object Tracking and Segmentation: A Unifying Approach](https://arxiv.org/pdf/1812.05050.pdf)|[CODE](https://github.com/foolwood/SiamMask)|
|Depth Estimation| [Digging Into Self-Supervised Monocular Depth Estimation](https://arxiv.org/pdf/1806.01260v3.pdf)|[CODE](https://github.com/nianticlabs/monodepth2)|
|Structure from Motion|  |[opensfm](https://github.com/mapillary/OpenSfM)|
|Image Generation|[LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS](https://arxiv.org/pdf/1809.11096.pdf)| |
|Face Generation| [StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)|[CODE](https://github.com/NVlabs/stylegan)|
|Image to Image| [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)|[CODE](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)|
|Style Transfer| [A Closed-form Solution to Photorealistic Image Stylization](https://arxiv.org/pdf/1802.06474v5.pdf)|[CODE](https://github.com/NVIDIA/FastPhotoStyle)|
|Keypoint Detection and Tracking| [SuperPoint: Self-Supervised Interest Point Detection and Description](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf)|[CODE](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)|
|Image Captioning| [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/pdf/1707.07998v3.pdf)|[CODE](https://github.com/facebookresearch/pythia)|
|Text to Image| [StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1710.10916.pdf)|[CODE](https://github.com/hanzhanggit/StackGAN)|


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
conda install -c conda-forge opencv
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



