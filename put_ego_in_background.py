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

# class Person:
#     id: int
#     img: np.ndarray
#     poses: list
#     segmented_img: np.ndarray

#     def __init__(self, id, img):
#         self.id = id
#         self.img = cv2.imread(img)
#         self.
#         self.poses = self.get_pose()
#         self.segmented_img = None



# class Frame:
#     id: int
#     background: np.ndarray
#     result_person: np.ndarray
#     original_person: np.ndarray
#     filled_background: np.ndarray
#     final_result: np.ndarray

#     def __init__(self, background, result_person, original_person, filled_background, final_result):
#         self.background = cv2.imread(background)
#         self.result_person = cv2.imread(result_person)
#         self.original_person = cv2.imread(original_person)
#         self.filled_background = cv2.imread(filled_background)
#     def __str__(self):
#         return f"background: {self.background.shape}, result_person: {self.result_person.shape}, original_person: {self.original_person.shape}, filled_background: {self.filled_background.shape}"
#     def 

i = 0

result_person_img = cv2.imread('./try_data_1/result_person/'+str(i)+'.jpg')
original_person_img = cv2.imread('./try_data_1/original_person/'+str(i)+'.jpg')
filled_background_img = cv2.imread('./try_data_1/filled_background/'+str(i)+'.jpg')
print(f"./try_data_1/result_person/'{str(i)}'.jpg")
cv2.imshow('result_person_img', result_person_img)
cv2.imshow('original_person_img', original_person_img)
cv2.imshow('filled_background_img', filled_background_img)

detector = poseDetector()
result_person_img1, result_person_annotated_image, result_person_masked_image, result_person_masked_image_,result_person_masked_image_eroded, result_person_masked_image_black = detector.findPose(result_person_img)

original_person_img1, original_person_annotated_image, original_person_masked_image, original_person_masked_image_,original_person_masked_image_eroded, original_person_masked_image_black = detector.findPose(original_person_img)

original_person_lmList, original_person_cxlist, original_person_cylist = detector.findPosition(original_person_img, draw=False)

result_person_lmList, result_person_cxlist, result_person_cylist = detector.findPosition(result_person_img, draw=False)

orig_r_ankle = original_person_lmList[28] # id, x, y
orig_l_ankle = original_person_lmList[27] # id, x, y

result_r_ankle = result_person_lmList[28] # id, x, y
result_l_ankle = result_person_lmList[27] # id, x, y

final_img = np.where(result_person_masked_image_black == 0, original_person_img1, result_person_img1)
cv2.imshow('final_img', final_img)
final_img  = cv2.resize(final_img, (176//3, 256//3))

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import match_template

result = match_template(filled_background_img, final_img)
ij = np.unravel_index(np.argmax(result), result.shape)
_, x, y= ij[::-1]
import ipdb; ipdb.set_trace()
umm_img = filled_background_img
cv2.circle(umm_img, (x, y), 10, (0, 0, 255), -1)
cv2.imshow('umm_img', umm_img)

cv2.waitKey(0)
