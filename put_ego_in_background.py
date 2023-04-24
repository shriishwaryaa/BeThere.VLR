'''
1. have an backgroung image
2. have the DGan generated person 

3. run mediapipe on Dgan result-> get segmentation of human 
4. run mediapipe on original img -> get the location of the human (feet?)
5. run background fill -> get the background

6. put the DGan reuslt human on the filled background on the right position 
7. do it for multiple frames and save it to a video

'''
import cv2
import mediapipe as mp
import time
import numpy as np
from module import poseDetector
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import match_template
import ipdb


i = 0

result_person_img = cv2.imread('./try_data_1/result_person/'+str(i)+'.jpg')
original_person_img = cv2.imread('./try_data_1/original_person/'+str(i)+'.jpg')
filled_background_img = cv2.imread('./try_data_1/filled_background/'+str(i)+'.jpg')
original_cuts_img = cv2.imread('./try_data_1/original_cuts/'+str(i)+'.jpg')

# print(f"./try_data_1/result_person/'{str(i)}'.jpg")
# cv2.imshow('result_person_img', result_person_img)
# cv2.imshow('original_person_img', original_person_img)
# cv2.imshow('filled_background_img', filled_background_img)
cv2.imshow('original_cuts_img', original_cuts_img)

# ipdb.set_trace()
detector = poseDetector()

result_person_img1, result_person_annotated_image, result_person_masked_image, result_person_masked_image_,result_person_masked_image_eroded, result_person_masked_image_black = detector.findPose(result_person_img)

original_person_img1, original_person_annotated_image, original_person_masked_image, original_person_masked_image_,original_person_masked_image_eroded, original_person_masked_image_black = detector.findPose(original_person_img)

original_cut_img1, original_cut_annotated_image, original_cut_masked_image, original_cut_masked_image_,original_cut_masked_image_eroded, original_cut_masked_image_black = detector.findPose(original_cuts_img)


original_person_lmList, original_person_cxlist, original_person_cylist = detector.findPosition(original_person_img, draw=False)

result_person_lmList, result_person_cxlist, result_person_cylist = detector.findPosition(result_person_img, draw=False)

original_cut_lmList, original_cut_cxlist, original_cut_cylist = detector.findPosition(original_cuts_img, draw=False)

orig_r_ankle = original_person_lmList[28] # id, x, y
orig_l_ankle = original_person_lmList[27] # id, x, y

result_r_ankle = result_person_lmList[28] # id, x, y
result_l_ankle = result_person_lmList[27] # id, x, y

oric_cut_r_ankle = original_cut_lmList[28] # id, x, y
oric_cut_l_ankle = original_cut_lmList[27] # id, x, y

ipdb.set_trace()

cv2.imshow('result_person_annotated_image', result_person_annotated_image)
cv2.imshow('original_cut_annotated_image', original_cut_annotated_image)
cv2.imshow('original_person_annotated_image', original_person_annotated_image)
# cv2.imshow('original_person_img1', original_person_img1)

final_img = np.where(result_person_masked_image_black == 0, original_person_img1, result_person_img1)
# cv2.imshow('final_img', final_img)
final_img  = cv2.resize(final_img, (176//3, 256//3))

result = match_template(filled_background_img, final_img)
# plt.imshow(result)
# plt.show()
# cv2.imshow('result', result)
ij = np.unravel_index(np.argmax(result), result.shape)
_, x, y= ij[::-1]
umm_img = filled_background_img
cv2.circle(umm_img, (x, y), 2, (0, 0, 255), -1)
cv2.rectangle(umm_img, (x, y), (x + final_img.shape[1], y + final_img.shape[0]), (0, 0, 255), 2)
cv2.imshow('umm_img', umm_img)

cv2.waitKey(0)
