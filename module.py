import cv2
import mediapipe as mp
import time
import math
import numpy as np
import os, os.path
import csv

class poseDetector():
    def __init__(self, static_image_mode=False, enable_segmentation=True, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = 2
        self.smooth_landmarks = smooth
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = True
        self.min_detection_confidence = detectionCon
        self.min_tracking_confidence = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
                                     self.static_image_mode, 
                                     self.model_complexity, 
                                     self.smooth_landmarks,
                                     self.enable_segmentation, 
                                     self.smooth_segmentation, 
                                     self.min_detection_confidence, 
                                     self.min_tracking_confidence
                                     )
    
    def findPose(self, img, draw=False):
        BG_COLOR = (0, 0, 0)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        kernel = np.ones((5, 5), np.uint8)
        kernel_black = np.zeros((5, 5), np.uint8)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, 
                                           self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        

        annotated_image = img.copy()
        masked_image = img.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((self.results.segmentation_mask,) * 3, axis=-1) > 0.1
        condition_masked = np.stack((self.results.segmentation_mask,) * 3, axis=-1) < 0.1
        masked_image_ = np.where(condition_masked, 255, 0).astype(np.uint8)
        masked_image_black = np.where(condition_masked, 0, 255).astype(np.uint8)
        # print(masked_image_, masked_image_.shape)
        # print(condition_masked)
        # condition_masked = cv2.erode(condition_masked, kernel, iterations=1)
        masked_image_eroded = cv2.erode(masked_image_, kernel, iterations=6)
        # masked_image_black = cv2.dilate(masked_image_black, kernel, iterations=6)

        
        bg_image = np.zeros(imgRGB.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        
        annotated_image = np.where(condition, annotated_image, bg_image)
        masked_image = np.where(condition_masked, masked_image, bg_image)
        masked_image_eroded = np.where(masked_image_eroded==255, masked_image, bg_image)

        # Draw pose landmarks on the image.
        self.mpDraw.draw_landmarks(
                                    annotated_image,
                                    self.results.pose_landmarks,
                                    self.mpPose.POSE_CONNECTIONS,
                                    )
        # cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

        # Plot pose world landmarks.
        # self.mpDraw.plot_landmarks(
        #                             self.results.pose_world_landmarks, 
        #                             self.mpPose.POSE_CONNECTIONS
        #                             )
        # print(annotated_image.shape)
        
        return img, annotated_image, masked_image, masked_image_, masked_image_eroded, masked_image_black
    
    def find_frame_with_background(self, masked_image):
        '''
        Takes in a frame and finds the frame in which there is a background in the reqired region
        Input
        1) image: the frame of interest
        Output
        1) frames_frames: list with 1's in index of frames with background   
        2) found_frames_list: list with frame numbers where background was found
        '''
        DIR = './data/frames_masked'
        list_names = [name for name in sorted(os.listdir(DIR)) if os.path.isfile(os.path.join(DIR, name))]
        found_frames_list=[]
        found_frames = []
        one = 1
        zero = 0
        condition = masked_image == (0,0, 255)
        c = np.sum(condition.astype(int), 2)
        c = np.where(c == 3, 1, 0)
        
        for i in range(1,len(list_names)+1):    
            img = cv2.imread('./data/frames_masked/masked' + str(i).zfill(3) + '.png')
            condition_img = img == (0,0, 255)
            c_img = np.sum(condition_img.astype(int), 2)
            c_img = np.where(c_img == 3, 1, 0)
            
            result = np.sum(np.multiply(c, c_img))

            if result > 0:
                # print("skipping image masked"+str(i).zfill(3) + '.png')
                found_frames.append(zero)
                continue
            else:
                found_frames.append(one)
                found_frames_list.append(i)

        return(found_frames, found_frames_list)


        

    def fill_background(self, image, frames_with_background):
        '''
        Input
        1) image: the frame in which background needs to be filled
        2) frames_with_background: list with 1's in index of frames with background

        Output
        1) filled_image: image with background filled in the region of interest
        '''
        condition = image == (0,0, 255)
        c = np.sum(condition.astype(int), 2)
        c = np.where(c == 3, 1, 0)
        # print(c.shape)
        cinv = np.where(c == 3, 0, 1)
        c =np.stack((c,c,c), axis=-1)
        # print(c.shape)
        cinv =np.stack((cinv,cinv,cinv), axis=-1)
        image_without_target = image*cinv
        image_without_target = image_without_target.astype(np.uint8)
        list_bg_images =[]

        j = frames_with_background[0]
        img = cv2.imread('./data/frames_masked/masked' + str(j).zfill(3) + '.png')
        filled_img = img*c
        filled_img = filled_img.astype(np.uint8)
        # for i in range(0,len(frames_with_background), 10):
        #     print(i)
        #     j = frames_with_background[i]
        #     img = cv2.imread('./data/frames_masked/masked' + str(j).zfill(3) + '.png')
        #     background_filled = img*c
        #     list_bg_images.append(background_filled.astype(np.uint8))
        # filled_img = np.mean(list_bg_images, axis=0).astype(np.uint8)
        final_img = (filled_img+image_without_target).astype(np.uint8)
        return final_img, image_without_target, list_bg_images, filled_img

        
    def findPosition(self, img, draw=True):
        self.lmList = []
        self.cxlist =[]
        self.cylist =[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                self.cxlist.append(cx)
                self.cylist.append(cy)
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList, self.cxlist, self.cylist
    # def findAngle(self, img, p1, p2, p3, draw=True):
    #     # Get the landmarks
    #     x1, y1 = self.lmList[p1][1:]
    #     x2, y2 = self.lmList[p2][1:]
    #     x3, y3 = self.lmList[p3][1:]
    #     # Calculate the Angle
    #     angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
    #                          math.atan2(y1 - y2, x1 - x2))
    #     if angle &lt; 0:
    #         angle += 360
    #     # print(angle)
    #     # Draw
    #     if draw:
    #         cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
    #         cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
    #         cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
    #         cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
    #         cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
    #         cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
    #         cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
    #         cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
    #         cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
    #                     cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    #     return angle

    def getpose(self, img, draw=True):
        self.results = self.pose.process(img)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, 
                                           self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img
def get_correct_crop(i, cxlist, cylist, img1):
    min_x = min(cxlist) - 10
    max_x = max(cxlist) + 10
    min_y = min(cylist) - 80
    max_y = max(cylist) + 50

    x_o = max_x - min_x
    y_o = (max_y - min_y)*(64/128)
    if x_o > y_o:
        padding = int(x_o * (128/64)/2 + 10)
        avg_y = (max_y + min_y)//2
        resized_image = img1[avg_y - padding:avg_y + padding,min_x:max_x,:]
    else:
        padding = int(y_o * (64/128)/2 + 10)
        avg_x =(max_x + min_x)//2
        resized_image = img1[min_y:max_y,avg_x - padding:avg_x+padding,:] 
    cropped_resised_img = cv2.resize(resized_image, (64,128))
    cv2.imwrite('./data/frames_cropped_resised/cropped' + str(i).zfill(3) + '.jpg', cropped_resised_img)
    
    return cropped_resised_img

def main():
    cap = cv2.VideoCapture('./data/IMG_1098.mp4')
    pTime = 0
    detector = poseDetector()
    i = 1
    
    with open('./data/files/poseforgans.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
    
        while True:
            success, img = cap.read()
            # img = cv2.imread('./data/result/'+str(i)+'.jpg')
            img1, annotated_image, masked_image, masked_image_,masked_image_eroded, masked_image_black = detector.findPose(img)
            
            lmList, cxlist, cylist = detector.findPosition(img1, draw=False)
            # cropped_resised_img = get_correct_crop(i, cxlist, cylist, img1)

            # cv2.imshow("Image", img1)
            # cv2.imshow("annotated_image", annotated_image)
            # cv2.imshow("masked_image", masked_image)
            # cv2.imshow("masked_image_", masked_image_)
            # cv2.imshow("masked_image_eroded", masked_image_eroded)
            # cv2.imshow("masked_image_black", masked_image_black)
            cv2.imshow("cropped resised image", cropped_resised_img)


            # cv2.imwrite('./data/frames_masked/masked' + str(i).zfill(3) + '.png', masked_image_eroded)
            # cv2.imwrite('./data/frames_with_pose/annotated' + str(i).zfill(3) + '.png', annotated_image)
            # cv2.imwrite('./data/frames_all_pose/annotated' + str(i).zfill(3) + '.png', img)
            # cv2.imwrite('./data/frames_black_white/mask' + str(i).zfill(3) + '.png', masked_image_black)
            # cv2.imwrite('./data/frames_cropped/cropped' + str(i).zfill(3) + '.png', cropped_image)
            # cv2.imwrite('./data/frames_cropped_resised/cropped' + str(i).zfill(3) + '.png', cropped_resised_img)
            i+=1
            cv2.waitKey(1)

if __name__ == "__main__":
    main()