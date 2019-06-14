# CnnSV-typer
Calling genotypes of deletions based on CUDA technology and bootstrapping algorithm
## Introduction
CnnSV-typer, an novel approach taking the two-dimensional images as inputs and calling the variation genotypes from the next-generation sequencing data through the deep learning network. CnnSV-typer is mainly divided into three parts: two-dimensional image generation, image compression and acceleration, and genotype calling based on CNN. In the first part, the gene text sequences of the relevant regions are extracted from BAM files according to candidate variations and are transformed into two-dimensional images according to the image transformation strategy. In the second part, the images are compressed by down-sampling accelerated by CUDA, providing consistent input with rich signals for the subsequent deep learning networks. In the third part, these compressed two-dimensional images and labels with known deletion genotypes are used to train the CNN. The final trained CNN model is than used to classify the genotypes of the images. The experimental results indicate that the proposed CnnSV-typer outperforms the current state-of-the-art methods on both simulation data and real population sequence data in both precision and sensitivity by up to 98.5% and 99.4%, respectively. The parallel acceleration technology of CUDA can improve the compression process by a speedup factor of 381.3. Meanwhile, CnnSV-typer can accurately call the structural variation genotypes of real data using bootstrapping algorithm.
## Requirements
  * python 3.6, numpy, Matplotlib
  * Cuda 8.0, Cudnn, pycuda
  * TensorFlow
  * Pysam
  * PIL
## Installation
### Tools
  bash Anaconda3-4.3.1-Linux-x86_64.sh <br/>
### Cuda & cudnn
   Installation tutorial can be downloaded from the official website  
### TensorFlow
* pip install tensorflow-gpu
### pysam
* pip install pysam
## Usage
### Data
BAM file & VCF file <br/>
First provide the bam files and vcf files, then extract the breakpoints of heterozygous deletion and homozygous deletiond from VCF files respectively, and then the non deletion regions are extracted by yourself.<br/>
### Generation Images of Candidates
Run the following program in the custom path <br/> 
* python 1.breakpoints_png_2.py
* python 2.breakpoints_png_1.py
* python 3.breakpoints_png_0.py
### Compress Images Using CUDA
* python 4.compress_png_cuda.py
### Split Data Set into Training Set and Test Set
* python 1.split_train_test.py img_path all_path/all_txt train_path/train_txt test_path/test_txt reset
### Training CNN
* python 2.cuda_normalization.py image_label_list file_dir X_train train_label train
* python 31.train_cnn.py X_train train_label
* python 32.train_cnn_boot.py X_train train_label noise list_lr list_batch list_epoch X_test test_label
### Using a trained network for calling genotype of deletions & Generating VCF File
* python 2.cuda_normalization.py image_label_list file_dir X_test test_label test
* python test_cnn.py X_test test_label
