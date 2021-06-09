import PIL as pl
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import time
from pathlib import Path 
import cv2
import matplotlib.pyplot as plt
import json

global train_folder
train_folder = "Sample Train"


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def _get_allFiles_inDirectory(folder_path):
    print("folder Path : ", folder_path)
    return os.listdir(folder_path)

## Filter blur images ##

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def _skipblur_images(imagePath,threshold=100):
    image = plt.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    if fm<threshold:return 0
    return fm

def _filter_bacground(img):
    #img = cv2.imread(img)
    gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    opening  = cv2.morphologyEx(thresh,cv2.MORPH_DILATE,kernel,iterations=3)
    opening  = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=3)

    # cnts = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
        # area = cv2.contourArea(c)
        # if area<50:
            # cv2.drawContours(opening,[c],-1,0,-1)
    # result = 255-opening
    result =cv2.GaussianBlur(opening,(1,1),0)
    result  = cv2.morphologyEx(result,cv2.MORPH_DILATE,kernel,iterations=3)
    cv2.cvtColor(result,cv2.COLOR_GRAY2RGB)
    cv2.addWeighted(result, 50., result, 0, 50.)
    #final = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    #print(result)
    
    
    
'''
img_dir : Image directory which will be recognized using Object Detection
annot_dir : x1<space>x2<space>x3<space>x4<space>y1<space>y2<space>y3<space>y4<space>word recognized using Object detection 
cropped_imgs_dir : directory to store images after running the function _trim_images
'''
def _trim_images(img_dir,annot_dir,cropped_imgs_dir,dumpFolder,counter):
  all_images = _get_allFiles_inDirectory(img_dir)
  all_images  = random.sample(all_images,len(all_images))
  step_counter,all_string_list,blurList = 0,[],[]
  print(len(all_images))
  
  for indd,e_image in enumerate(tqdm(all_images)):
    
    file_name = e_image.replace('.jpg','.txt')
    orgi_annot = os.path.join(annot_dir,file_name)
    orgi_annot = r'{}'.format(orgi_annot)
    with open(orgi_annot,'r',encoding="cp437",errors='ignore') as ff:
      all_lines = ff.readlines()
    ff.close()
    if all_lines in [None,[]]:continue
    for e_line in all_lines:
      split_list = e_line.split(' ')
      px = [float(e_str) for e_str in split_list[0:4]]
      width = max(px) - min(px)
      py = [float(e_str) for e_str in split_list[4:8]]
      ht = max(py) - min(py)
      box_coords = px+py
      im = Image.open(img_dir+'/'+str(e_image))
      crop_image = im.crop((px[0]+.75,py[0]+.75,px[2]+.75,py[2]+.75))
      data = np.asarray(crop_image)
      
      try:
        data[0]
      except:
        continue
      if data.shape[0]/data.shape[1]<0.1 or data.shape[0]/data.shape[1]>10:
        continue

      resize_img = cv2.resize(data,(128,128),fx=1,fy=2,interpolation=cv2.INTER_CUBIC)
      data = np.asarray(resize_img)
      _filter_bacground(data)
      # cv2.imshow("resize_img",resize_img)
      # cv2.waitKey()
      new_img_path = cropped_imgs_dir+'/'+str(counter)+".jpg"
      new_img_path = r'{}'.format(new_img_path)
      cv2.imwrite(new_img_path,data)
      try:
        blurVal = _skipblur_images(new_img_path,300)
      except:continue
      if blurVal==0:
        os.remove(new_img_path)
        continue
      blurList.append(str(counter)+"-->"+str(blurVal)+'\n')
      child_path = os.path.join(train_folder,str(counter)+".jpg")
      final_str = str.join("",[child_path,'\t',str(split_list[-1]),'\t',
                                os.path.join(img_dir+e_image),'\t',str(px+py)])
      all_string_list.append(final_str)
      counter+=1
      step_counter+=1
      if counter == 500:
        return counter,all_string_list
        
      if step_counter ==1000:
        return counter,all_string_list
    if indd==len(all_images)-50:
        return counter,all_string_list




def _trim_validation_images(img_dir,boxFile,cropped_imgs_dir):
  all_images = _get_allFiles_inDirectory(img_dir)
  with open(boxFile,'r') as fh:
    infoDict = fh.load(fh)
  
  for e_image,posList in infoDict.items():
      px = posList[0:4]
      width = max(px) - min(px)
      py = posList[4:8]
      ht = max(py) - min(py)
      box_coords = px+py
      im = Image.open(img_dir+'/'+str(e_image))
      crop_image = im.crop((px[0]+.75,py[0]+.75,px[2]+.75,py[2]+.75))
      data = np.asarray(crop_image)

      resize_img = cv2.resize(data,(128,128),fx=1,fy=2,interpolation=cv2.INTER_CUBIC)
      data = np.asarray(resize_img)
      _filter_bacground(data)

      new_img_path = cropped_imgs_dir+'/'+str(e_image)+".jpg"
      new_img_path = r'{}'.format(new_img_path)
      cv2.imwrite(new_img_path,data)
 


    


def main():
    dirpath = Path().absolute()
    parent_img_path = os.path.join(dirpath,"Synthetic Train Set - Detection & Recognition","Images")
    parent_annot_path = os.path.join(dirpath,"Synthetic Train Set - Detection & Recognition","Annotation")
    ## To create folder to add all cropped images for training ##
    cropped_imgs_dir = os.path.join(dirpath,"ImgtoText","Train","Sample Train")
    dumpFolder = os.path.join(dirpath,"ImgtoText","Train","Dump")
    blurInfo = os.path.join(dirpath,"ImgtoText","Train")
    blurInfo_path = os.path.join(blurInfo,"blurInfo.txt")
    blurInfo_path = r'{}'.format(blurInfo_path)
    print("cropped_imgs_dir",cropped_imgs_dir)
    try:    
        os.makedirs(cropped_imgs_dir)
    except:
        pass
    
    ## To create a foder to store a txt file for annotations ##
    try:
        cropped_annots_dir = os.path.join(dirpath,"ImgtoText","Train","Annotations1")
        os.makedirs(cropped_annots_dir)
    except:
        pass
    txt_file_name  = "train_crop_annot.txt"
    txt_file_path = os.path.join(cropped_annots_dir,txt_file_name)
    txt_file_path = r'{}'.format(txt_file_path)
    print("txt_file_path",txt_file_path)
    
    
    full_list = list(range(1,26))
    full_list = random.sample(full_list,len(full_list))
    print(full_list)
    print("Entering for loop")
    all_annot_info,all_blur_info = [],[]
    counter = 0
    for i in full_list:
      i =str(i)

      counter,all_string_list = _trim_images(os.path.join(parent_img_path,i),os.path.join(parent_annot_path,i),cropped_imgs_dir,dumpFolder,counter)
      
      all_annot_info.extend(all_string_list)
      print("********Images cropped **************** ----> {}".format(counter))
      
      
      if counter > 5000:
        print("Finishing Cropping Images")
        break

    print(len(all_annot_info))
    with open(txt_file_path,'w',encoding="cp437",errors='ignore') as fh:
        fh.writelines(all_annot_info)
    fh.close()

        
        
main()

#_skipblur_images(r"D:\Padhai Work assignments\Capstone_Project\ImgtoText\Train\TrainImages\30.jpg")
