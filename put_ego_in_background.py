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


i = 323
while True:
    result_person_img = cv2.imread('./try_data_1/result_person_ishu/'+str(i)+'.jpg')
    # original_person_img = cv2.imread('./try_data_1/original_person/'+str(i)+'.jpg')
    # filled_background_img = cv2.imread('./try_data_1/filled_background/'+str(i)+'.jpg')
    # original_cuts_img = cv2.imread('./try_data_1/original_cuts/'+str(i)+'.jpg')
    background_img = cv2.imread(f'./try_data_1/onlybackground/{i:05d}.png')
    background_img = cv2.resize(background_img, (176, 256))

    # print(f"./try_data_1/result_person/'{str(i)}'.jpg")
    # cv2.imshow('result_person_img', result_person_img)
    # cv2.imshow('original_person_img', original_person_img)
    # cv2.imshow('filled_background_img', filled_background_img)
    # cv2.imshow('original_cuts_img', original_cuts_img)

    # ipdb.set_trace()
    detector_result_person = poseDetector()
    # detector_original_person = poseDetector()
    # detector_original_cut = poseDetector()

    result_person_img1, result_person_annotated_image, result_person_masked_image, result_person_masked_image_,result_person_masked_image_eroded, result_person_masked_image_black = detector_result_person.findPose(result_person_img)

    # original_person_img1, original_person_annotated_image, original_person_masked_image, original_person_masked_image_,original_person_masked_image_eroded, original_person_masked_image_black = detector_original_person.findPose(original_person_img)

    # original_cut_img1, original_cut_annotated_image, original_cut_masked_image, original_cut_masked_image_,original_cut_masked_image_eroded, original_cut_masked_image_black = detector_original_cut = detector_result_person.findPose(original_cuts_img)


    # original_person_lmList, original_person_cxlist, original_person_cylist = detector_result_person.findPosition(original_person_img, draw=False)

    # result_person_lmList, result_person_cxlist, result_person_cylist = detector_original_person.findPosition(result_person_img, draw=False)

    # original_cut_lmList, original_cut_cxlist, original_cut_cylist = detector_original_cut = detector_result_person.findPosition(original_cuts_img, draw=False)

    # orig_left_shoulder = original_person_lmList[11] # id, x, y
    # orig_right_shoulder = original_person_lmList[12] # id, x, y
    # orig_left_hip = original_person_lmList[23] # id, x, y
    # orig_right_hip = original_person_lmList[24] # id, x, y


    # cut_left_shoulder = original_cut_lmList[11] # id, x, y
    # cut_right_shoulder = original_cut_lmList[12] # id, x, y
    # cut_left_hip = original_cut_lmList[23] # id, x, y
    # cut_right_hip = original_cut_lmList[24] # id, x, y

    # orig_y = (abs(orig_left_shoulder[1] - orig_right_shoulder[1]) + abs(orig_left_hip[1] - orig_right_hip[1]))/2
    # orig_x = (abs(orig_left_hip [2]- orig_left_shoulder[2]) + abs(orig_right_hip[2] - orig_right_shoulder[2]))/2

    # cut_y = (abs(cut_left_shoulder[1] - cut_right_shoulder[1]) + abs(cut_left_hip[1] - cut_right_hip[1]))/2
    # cut_x = (abs(cut_left_hip[2] - cut_left_shoulder[2]) + abs(cut_right_hip[2] - cut_right_shoulder[2]))/2

    # x_ratio_cut2orig = abs(orig_x/cut_x)
    # y_ratio_cut2orig = abs(orig_y/cut_y)

    # x_ratio_orig2cut = 1/x_ratio_cut2orig
    # y_ratio_orig2cut = 1/y_ratio_cut2orig

    # print('x_ratio_cut2orig', x_ratio_cut2orig)
    # print('y_ratio_cut2orig', y_ratio_cut2orig)

    # print('x_ratio_orig2cut', x_ratio_orig2cut)
    # print('y_ratio_orig2cut', y_ratio_orig2cut)

    # resized_shape = (int(original_person_img1.shape[1]*y_ratio_cut2orig), int(original_person_img1.shape[0]*x_ratio_cut2orig))

    # print('resized_shape', resized_shape)
    # hi = cv2.resize(original_person_img1, resized_shape)
    # cv2.imshow('original_person_img1', original_person_img1)


    # ipdb.set_trace()

    # cv2.imshow('result_person_annotated_image', result_person_annotated_image)
    # cv2.imshow('original_cut_annotated_image', original_cut_annotated_image)
    # cv2.imshow('original_person_annotated_image', original_person_annotated_image)
    # cv2.imshow('original_person_img1', original_cut_img1)
    # cv2.imshow('original_cut_annotated_image', original_cut_annotated_image)

    final_img = np.where(result_person_masked_image_black == 0, background_img, result_person_img1)
    # final_img = np.where(result_person_masked_image_black == 0, original_person_img1, np.zeros_like(original_person_img1))
    # cv2.imshow('final_img', final_img)
    cv2.imwrite('./try_data_1/final_img_ishu/' + str(i)+ '.jpg', final_img)
    # original_person_img1  = cv2.resize(original_person_img1, (176//3, 256//3))

    # result = match_template(filled_background_img, original_person_img1)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # ipdb.set_trace()

    # main_gray = cv2.cvtColor(filled_background_img, cv2.COLOR_BGR2GRAY)
    # patch_gray = cv2.cvtColor(original_person_img1, cv2.COLOR_BGR2GRAY)

    # res = cv2.matchTemplate(filled_background_img, original_person_img1, cv2.TM_CCOEFF_NORMED)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # top_left = max_loc
    # bottom_right = (top_left[0] + original_person_img1.shape[1], top_left[1] + original_person_img1.shape[0])

    # filled_background_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = original_person_img1
    # cv2.imshow('Result', filled_background_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    # plt.imshow(result)
    # plt.show()
    # # cv2.imshow('result', result)
    # ij = np.unravel_index(np.argmax(result), result.shape)
    # _, x, y= ij[::-1]
    # umm_img = filled_background_img
    # cv2.circle(umm_img, (x, y), 2, (0, 0, 255), -1)
    # cv2.circle(umm_img, (x, y), 2, (0, 255, 0), -1)
    # cv2.circle(umm_img, (x, y), 2, (0, 255, 0), -1)
    # cv2.circle(umm_img, (x, y), 2, (0, 255, 0), -1)
    # cv2.circle(umm_img, (x, y), 2, (0, 255, 0), -1)

    # cv2.rectangle(umm_img, (x, y), (x + original_person_img1.shape[1], y + original_person_img1.shape[0]), (0, 0, 255), 2)
    # cv2.imshow('umm_img', umm_img)

    # cv2.waitKey(0)
    i +=1
