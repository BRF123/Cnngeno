# Cnngeno
A high-precision deep leaning based strategy for the calling of structural variation genotype
## Introduction
The current hand-crafted features and parametric statistical models used for genotype calling are still greatly influenced by the size of structural variations and the coverage of the next sequencing data. This paper attempts to bridge this gap by proposing a new calling approach based on deep learning, namely Cnngeno. Cnngeno converts sequencing texts to their corresponding image datas and classifies the genotypes of the image datas by Convolutional Neural Network (CNN). Moreover, the convolutional bootstrapping algorithm is adopted, which greatly improves the anti-noisy label ability of the deep learning network on real data. In comparison with current tools, including Pindel, LUMPY+SVTyper, Delly, CNVnator and GINDEL, Cnngeno  achieves a peak precision and sensitivity of 100% respectively and a wider range of detection lengths on various coverage data. Besides, Cnngeno outperforms another state-of-the-art CNN based method for the calling of structural variation genotype by better performance on real data. The experimental results suggest that Cnngeno has significant implications for future research on genotype calling.
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
* python 4.compress_png.py
### Split Data Set into Training Set and Test Set
* python 1.split_train_test.py img_path all_path/all_txt train_path/train_txt test_path/test_txt reset
### Training CNN
* python 2.cuda_normalization.py image_label_list file_dir X_train train_label train
* python 3.train_cnn_boot.py X_train train_label noise list_lr list_batch list_epoch X_test test_label
### Using a trained network for calling genotype of deletions & Generating VCF File
* python 2.cuda_normalization.py image_label_list file_dir X_test test_label test
* python 4.test_cnn.py X_test test_label
