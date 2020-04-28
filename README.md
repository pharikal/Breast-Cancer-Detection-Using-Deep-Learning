# Breast-Cancer-GANs
**CSC7991: Introduction to Deep Learning**
Final Project Report


# **Nuclei Detection on Histopathology Images to identify Breast Cancer**

**Supervised by**
Dr. Ming Dong

**Submitted by:** 

**Group 7** 

| _Debadeep Pharikal_ | _Bipasha Banerjee_ | _Alokparna Bandyopadhyay_ |


# ABSTRACT

This course project is based on Nuclei Detection on Histopathology Images to identify Breast Cancer using Deep Learning. The major goal of this course project is to experiment deep learning techniques covered under CSC 7991: Introduction to Deep Learning and work towards implementation of those knowledge to develop a near to perfect model in predicting Breast Cancer. The more accurate the models are, more chances of artificial systems to predict if the person is having Breast Cancer. The process of Nuclei detection in high-grade breast cancer images is quite challenging in the case of image processing techniques due to certain heterogeneous characteristics of cancer nuclei such as enlarged and irregularly shaped nuclei. The visual attributes of cells, such as the nuclear morphology, are critical for histopathology image analysis.__Based on the proposed cell-level visual representation learning, we further develop a pipeline that exploits the varieties of cellular elements to perform histopathology image classification._


# Introduction:


## **Overview**

Breast Cancer is a group of disease in which cells in breast tissue change and divide uncontrollably leading to lump or mass. It is the most common type of cancer which causes 411,000 annual deaths worldwide. After skin cancer, breast cancer is the most common cancer diagnosed in women in the United States and still open area for research especially, the area digital image analysis. Hence the major motivation of this course project is to develop an effective product in medical domain that extracts features from histopathology images and helps in identification of breast cancer.

Supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications but proven to be computationally extensive. Our implementation of this product covers conventional CNN approach to Generative adversarial networks (GAN) for nuclei-detection of histopathology images. GAN models have evolved over time. Thus, our further experiment contains multiple variations of GANs. The results are compared with respect to accuracy and performance, to find best deep learning model for our problem.

 
## **Literature Review** 

   
### **Convolutional Neural Networks (CNN)**

Convolutional neural networks (CNN) is a special type of artificial neural network architecture with some features of the visual cortex and works phenomenally well on computer vision tasks like image classification, object detection, image recognition, etc. They are composed of many &quot;filters&quot;, which convolve, or slide across the data, and produce an activation at every slide position. But when it comes to data synthesizing and image-to-image translation, CNNs do not work that well. GANs typically works best as image synthesizers and typically use CNN architectures for their generator and discriminator models.

   
### **Generative Adversarial Networks (GAN)**

Generative adversarial networks (GANs) are algorithmic architectures that use two neural networks, pitting one against the other (thus the &quot;adversarial&quot;) in order to generate new, synthetic or fake instances of data that can pass for real data. The model consists of a generative model and a discriminative model – both realized as multilayer perceptrons (MLPs). In this architecture a generative model G captures the data distribution, and its adversary a discriminative model D estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player.

  
### **Conditional Generative Adversarial Networks (CGAN)**

The conditional generative adversarial network, or cGAN is a variation of GAN which involves the conditional generation of images by a generator model. Image generation can be conditional on a class label, in case available, allowing the targeted generation of images of a given domain.

  
### **Deep Convolutional Generative Adversarial Networks (DCGAN)**

Supervised learning with convolutional networks (CNNs) has seen huge success in analyzing histopathological applications. In comparison to that unsupervised learning with CNN has not been successful. The idea of DCGAN help to mitigate that problem. A class of CNNs also known as DCGANs, which have certain architectural constraints, shows that they are suitable for unsupervised learning. We extend this architecture to make it supervised such that it can solve our classification with less computational cost.


### **Semi Supervised Generative Adversarial Networks (SGAN)**

SGAN is an architecture which is based on semi-supervised context by forcing D to output N+1 different output class, N different &quot;real&quot; classes, and an additional fake class (anything that came from G). In our case, N=2 (real nuclei, and real non-nuclei).


### **Least Squares Generative Adversarial Networks (LSGAN)**

Regular GANs hypothesize the discriminator as a classifier with the sigmoid cross entropy loss function. This loss function may lead to the vanishing gradients problem during the learning process. To overcome such a problem, the conventional loss function for discriminator is replaced with least squares loss function. There are two benefits of LSGANs over regular GANs. First, LSGANs can generate higher quality images than regular GANs. Second, LSGANs perform more stable during the learning process.


### **Coupled Generative Adversarial Networks (CoGAN)**

COGAN consists of pair of GANs. Ming et al proposed CoGAN architecture can learn a joint distribution without any tuple of corresponding images. It can learn a joint distribution with just samples drawn from the marginal distributions. This is achieved by enforcing a weight-sharing constraint that limits the network capacity and favors a joint distribution solution over a product of marginal distributions one.

**Objective** 

The major motivation of this course project is to develop an effective product using Deep Learning method especially in the field of GANs and provide the comparable accuracy of the literature study done. Our novel contribution is to build all GAN models using supervised or semi-supervised learning methods and to choose best GAN model for classification of histopathology images in predicting breast cancer. Additionally, we will change some of the architectural constraints to fine tune the model accuracy and make the model suitable for supervised/semi-supervised learning. We propose additional momentum which stabilized most of our training procedure.

**Dataset** 

The original data set contains 537 hematoxylin–eosin (H &amp; E) stained histopathological images obtained from digitized glass slides. H&amp;E stained breast histopathology glass slides were scanned into a computer using a high-resolution whole slide scanner, Aperio ScanScope digitizer, at 40x optical magnification.

The dataset corresponds to 49 lymph node-negative and estrogen receptor positive breast cancer (LN-, ER+ BC) patients at **Case Western Reserve University**. For sake of running the models with our existing configurations we borrowed the dataset used by Xu et al which contains less amount samples.

1. The training data includes 2,000 nuclear and 6,000 non-nuclear patches.
2. There are 1,000 patches for validation, 500 nuclear patches and 500 non-nuclear.
3. This dataset already contains the data divided into train and validation. The value range of each image was originally [0…1]. but we normalize it to be [-1…1]. We modify training and testing labels. 0 represents non-nucleus, 1 represents nucleus.

The dataset can be obtained from 
link: [https://engineering.case.edu/centers/ccipd/data](https://engineering.case.edu/centers/ccipd/data)

This course project explores, and reviews various deep learning techniques used for histopathology image analysis with a goal on breast cancer detection. We compared multiple GAN models and showed how an efficient deep-learning model can capture high-level feature representations of pixel intensity in a supervised and semi-supervised manner. LSGAN stands out the best for a supervised learning task owing to its high AveP and SGAN comes handy when we have less labelled data compared to Supervised GAN models. So, we conclude that these high-level features enable the classifier to work very efficiently for detecting multiple nuclei from a large cohort of histopathological images as well as to generate realistic synthesized representations of nuclei and non-nuclei images. This review aims at complementing the effort of pathologists, in examining and analyzing biopsy samples, by computer aided techniques and thereby help medicine and science to predict breast cancer.

# Acknowledgement

We thank professor Dr. Ming Dong for his guidance with the concepts of Deep Learning &amp; his insights regarding the approach of solving a classification task like this and teaching assistant Mr. Qisheng He for his valuable inputs with Keras &amp; Tensorflow.

# References

1. I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, &amp; Y. Bengio. (2014, Jun.). &quot;Generative Adversarial Networks.&quot; ArXiv E-Prints. [On-line]. Available: [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661) [7 May 2018].
2. A. Radford, L. Metz, &amp; S. Chintala. (2016, Jan.). &quot;Unsupervised representation learning with deep convolutional generative adversarial networks.&quot; ArXiv E-Prints. [On-line]. Available: [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434) [7 Jan 2016].
3. Unsupervised Learning for Cell-level Visual Representation in Histopathology Images with Generative Adversarial Networks Bo Hu] , Ye Tang] , Eric I-Chao Chang, Yubo Fan, Maode Lai and Yan Xu\*.[https://arxiv.org/pdf/1711.11317.pdf](https://arxiv.org/pdf/1711.11317.pdf)
4. V. Vargas &amp; J. Koller. &quot;GAN-for-Nuclei-Detection.&quot; Internet: [https://github.com/vmvargas/GAN-for-Nuclei-Detection](https://github.com/vmvargas/GAN-for-Nuclei-Detection) , May 7, 2018 [7 May 2018].
5. J. Xu, L. Xiang, Q. Liu, H. Gilmore, J. Wu, J. Tang, &amp; A. Madabhushi, &quot;Stacked sparse autoencoder (SSAE) for nuclei detection on breast cancer histopathology images,&quot; in IEEE Transactions on Medical Imaging, Vol. 35 no. 1, pp. 119-130, July 2016. [https://europepmc.org/article/pmc/pmc4729702](https://europepmc.org/article/pmc/pmc4729702)
6. Keras implementation of General Adversarial Network - [https://github.com/eriklindernoren/Keras-GAN](https://github.com/eriklindernoren/Keras-GAN)
7. Coupled GAN code by authors - [https://github.com/mingyuliutw/CoGAN](https://github.com/mingyuliutw/CoGAN)
8. Conditional GAN - [https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/](https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/)
