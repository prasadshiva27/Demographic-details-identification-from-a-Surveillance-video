# Demographic-details-identification-from-a-Surveillance-video
This repo helps in identifying the Age and Gender of persons in a Surveillance Video usually from far view camera video,
where there is a difficulty in detetcting face and where most exisitng alogorithms use face for identifying Age and Gender.

This complete repo used DeepSort for object tracking and YOLOv4 for Object Detetction and Custom trainind NN for 
Identifaction of Age and Gender details.

### Approach
First Persons are identified in the video and thier tracking is observed and after they reach about half the screen I cropped
them and sent to my trained model which identified the Age and Gender.

Later I capture these deatails in AWS dynamo DB and make some analysis using the demographics. This work is still in progress.

All he work is done in Google Colab using the GPU.

### Demo
[![Watch the video](https://img.youtube.com/vi/fdTLaSy_NKc/0.jpg)](https://youtu.be/fdTLaSy_NKc)

This identifiaction from far view images task is a multi label single class classification problem and is approached with a 
specialized convolutional neural network model that achieves an overall accuracy of 82.53%% for 7 attributes namely 
Male, Female, PersonAgeLess15, PersonAgeLess30, PersonAgeLess45, PersonAgeLess60 and PersonAgeLarger60 as defined in the 
[PETA](http://mmlab.ie.cuhk.edu.hk/projects/PETA.html) dataset which consists of far-view images of pedestrians.
