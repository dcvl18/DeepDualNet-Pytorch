# Face Anti-Spoofing Using Deep Dual Network

"Face anti-spoofing using deep dual network," is accepted in IEIE Transactions on Smart Processing and Computing, accepted, Mar. 2020, written by Yongjae Gwak, Chanho Jeong, J-H. Roh, S. Cho, and Wonjun Kim*

## Overview

An implementation of DeepDualNet using Pytorch

## Requirment

* torch==1.1.0
* torchvision==0.3.0
* opencv-python

## Dataset

We construct the dataset using a stereo camera with 50 subjects by ourselves, but unfortunately we cannot open it in public. Instead of sharing all of it, we offer one subject to test the model.

## Train and Test

You can train and test the model by "python train.py". After training the model, we immediately test the trained model.

## The proposed DeepDualNet for face anti-spoofing

<p align="center">
<img src="https://user-images.githubusercontent.com/58552068/78256268-43d47380-7533-11ea-94fc-ed5028101f76.jpg" />
</p>

## Experimental Results

|  <center>Method</center> |  <center>Number of subjects</center> |  <center>EER</center> |
|:--------|:--------:|--------:|
|**X. Sun et al.** | <center>35</center> |*0.68* |
|**DeepDualNet** | <center>35</center> |*0.51* |
|**DeepDualNet** | <center>50</center> |*0.48* |

Compared to this method of which EER is 0.68 based on the dataset that total 35 subjects participated in, the proposed deep dual network yields more reliable performance, i.e., EER=0.48.

## The examples of our constructed dataset
<p align="center">
<img src="https://user-images.githubusercontent.com/58552068/70987797-f4914100-2103-11ea-8f81-7dbf3ec12540.png" />
</p>


## The examples of face anti-spoofing
<p align="center">
 
<img src="https://user-images.githubusercontent.com/58552068/70986583-877cac00-2101-11ea-843c-7bda09c5e107.png" />
</p>

## Implementation

The examples of demonstrations based on a single laptop PC with i7-8750H@2.20GHz CPU and NVIDIA GeForce GTX 1060 

<p align="center">
<img src="https://user-images.githubusercontent.com/58552068/70986341-0e7d5480-2101-11ea-89bf-d51c5a9b0340.png" />
</p>
