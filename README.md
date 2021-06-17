# Capstone_Project
M/C Transliteration project (Image as Input)

Task 1 : Object Detection with pre-trained Detectron2 model
Predicting boxes having Texts in an Image

https://github.com/abhiraman/Capstone_Project/blob/main/Text_Detection_Detectron2.ipynb has the script for object detection training
Directory - Demo Train Data for Object Detection  ---> contains demo training images and corresponding their annotations
References : Many online codes available for training using Detectron2 

![image](https://user-images.githubusercontent.com/38239676/122356970-eb8b6500-cf70-11eb-944f-831212c67a35.png)

Task 2 : Creating training & validation images for Image to Text Recognition (Image -> Hindi Word) 
Sub task 1 : Cropping the object detected text and resizing & some bit of image clean up done using the below code 
https://github.com/abhiraman/Capstone_Project/blob/main/utils/Create_TrainData_for_ImgtoText.py

Sub task 2 :  Image to Text Recognition
For training, cropped object detected text images & their true word labels are used. True word labels are got from Task 1 annotations 
https://github.com/abhiraman/Capstone_Project/blob/main/Img2text/Image_to_text.ipynb
References : https://github.com/GitYCC/crnn-pytorch
http://arxiv.org/abs/1507.05717 ---> Model very similar to this paper is implemented

![image](https://user-images.githubusercontent.com/38239676/122358584-65701e00-cf72-11eb-85c4-0f092282a78f.png)

***************Predicted Hindi Word ---->पाँचसौ***************
![image](https://user-images.githubusercontent.com/38239676/122358630-7325a380-cf72-11eb-86e0-d82751b16589.png)

***************Predicted Hindi Word ---->अरे***************
![image](https://user-images.githubusercontent.com/38239676/122358664-7b7dde80-cf72-11eb-8991-1809302814fa.png)

***************Predicted Hindi Word ---->सम्भा***************
![image](https://user-images.githubusercontent.com/38239676/122358733-89336400-cf72-11eb-971a-e42acfb49413.png)

***************Predicted Hindi Word ---->परिसर***************
![image](https://user-images.githubusercontent.com/38239676/122358751-8fc1db80-cf72-11eb-973f-4ec8bba358d8.png)

***************Predicted Hindi Word ---->रुपये***************

Task 3 : M/C Transliteration with Attention model ----> Convert Hindi word To English word
https://github.com/abhiraman/Capstone_Project/blob/main/Transliteration%20Model/M_C_Transliteration.ipynb
References : GUVI deep learning course lecture videos 
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

पाँचसौ   Transliterated to ---->     PANCCAU
अरे   Transliterated to ---->     ARE
सम्भा   Transliterated to ---->     SAMBHA
परिसर   Transliterated to ---->     PARISER
रुपये   Transliterated to ---->     RUPAYE


Task 4 : Clubbing three models , Object Detection -> Image 2 text Recognition -> M/C transliteration for validation 
Inference run of the project by clubbing all the 3 models by  pre- loading trained weights
https://github.com/abhiraman/Capstone_Project/blob/main/Clubbed_Models.ipynb

NOTE: Further training will further improve results 

