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
        BG_COLOR = (0, 0, 255)
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
        masked_image_ = cv2.erode(masked_image_, kernel, iterations=6)
        masked_image_black = cv2.dilate(masked_image_black, kernel, iterations=6)

        
        bg_image = np.zeros(imgRGB.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        
        annotated_image = np.where(condition, annotated_image, bg_image)
        masked_image = np.where(condition_masked, masked_image, bg_image)
        masked_image_eroded = np.where(masked_image_==255, masked_image, bg_image)
        # print(masked_image)
        # Draw pose landmarks on the image.
        # self.mpDraw.draw_landmarks(
        #                             annotated_image,
        #                             self.results.pose_landmarks,
        #                             self.mpPose.POSE_CONNECTIONS,
        #                             )
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
        # print (len(list_names), list_names)
        found_frames_list=[]
        found_frames = []
        one = 1
        zero = 0
        condition = masked_image == (0,0, 255)
        c = np.sum(condition.astype(int), 2)
        c = np.where(c == 3, 1, 0)
        # print(f"c: {np.sum(c)}")

        # print(c.shape, c)
        
        for i in range(1,len(list_names)+1):    
            img = cv2.imread('./data/frames_masked/masked' + str(i).zfill(3) + '.png')
            condition_img = img == (0,0, 255)
            c_img = np.sum(condition_img.astype(int), 2)
            c_img = np.where(c_img == 3, 1, 0)
            # print(f"c_img: {np.sum(c_img)}")

            # print(c_img.shape, c_img)
            
            result = np.sum(np.multiply(c, c_img))

            if result > 0:
                # print("skipping image masked"+str(i).zfill(3) + '.png')
                found_frames.append(zero)
                continue
            else:
                # print("                               found background in frame masked"+str(i).zfill(3) + '.png')
                found_frames.append(one)
                found_frames_list.append(i)

            # mask1 = condition[:, :, 0].astype(int)
            # print(mask1)
            # print(condition)
            # a = img*condition
            # print("                                              ",a.reshape(-1).shape)
            # condition2= img == (0,0,255)
            # print(condition.shape, condition2.shape)
            # combine = condition.all() and condition2.all()
            # if combine.any() == True:
            #     print("skipping image masked"+str(i).zfill(3) + '.png')
            #     found_frames.append(zero)
            #     continue
            # else:
            #     print("                               found background in frame masked"+str(i).zfill(3) + '.png')
            #     found_frames.append(one)
            #     found_frames_list.append(i)





            # count = 0
            # for i in range(img.shape[0]*img.shape[1]*img.shape[2]):
            #     if (a.reshape(-1)[i] == [0,0,255]).all():
            #         count +=1
            # print(count)
            
                
            # for l in range(a.shape[0]):
            #     for k in range(a.shape[1]):
            #         if (a[l][k] == [0,0,255]).all():
            #             count +=1
            #             # print("skipping image masked"+str(i).zfill(3) + '.png')
            #             # found_frames.append(zero)
            #             # continue
            #         # else:
            #             # print("                               found background in frame masked"+str(i).zfill(3) + '.png')
            #             # found_frames.append(one)
            #             # found_frames_list.append(i)
            # print(count)
            # if count ==0:
            #     print("                               found background in frame masked"+str(i).zfill(3) + '.png')
            #     found_frames.append(one)
            #     found_frames_list.append(i)
            # else:
            #     print("skipping image masked"+str(i).zfill(3) + '.png')
            #     found_frames.append(zero)

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
        print(c.shape)
        cinv = np.where(c == 3, 0, 1)
        c =np.stack((c,c,c), axis=-1)
        print(c.shape)
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
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                self.cxlist.append(cx)
                self.cylist.append(cy)
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList, self.cxlist, self.cylist
    '''
    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle &lt; 0:
            angle += 360
        # print(angle)
        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
    ''' 
def main():
    cap = cv2.VideoCapture('./data/IMG_1098.mp4')
    pTime = 0
    detector = poseDetector()
    i = 1
    
    with open('./data/files/poseforgans.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
    
        while True:
            success, img = cap.read()
            img, annotated_image, masked_image, masked_image_,masked_image_eroded, masked_image_black = detector.findPose(img)
            # if i%50 == 0:   
            #     print("getting frames for background ............")
            #     frames_list, frame_with_background = detector.find_frame_with_background(masked_image_eroded)
            #     print("##########################  number of background frames is  ###################################")
            #     print( len(frame_with_background))
            #     final_img, image_without_target, list_bg_images, filled_img = detector.fill_background(masked_image_eroded , frame_with_background)
            #     cv2.imshow("filled image", final_img)
            #     cv2.imshow("image without target", image_without_target)
            #     cv2.imshow("filled_img", filled_img)
            #     # cv2.imshow("list_bg_images", list_bg_images[0])
            #     cv2.imwrite('./data/filled_background/filled' + str(i).zfill(3) + '.png', final_img)

            
            lmList, cxlist, cylist = detector.findPosition(img, draw=False)
            print(lmList)
            print(cxlist)
            print(cylist)
            
            min_x = min(cxlist)
            max_x = max(cxlist)
            min_y = min(cylist)
            max_y = max(cylist)
            
            print(min_x)
            print(min_y)
            print(max_x)
            print(max_y)
            box = ()
            
            cropped_image = annotated_image[min_y-50:max_y+20,min_x-30:max_x+20,:]
            # cropped_image = annotated_image[min_y:max_y][min_x:max_x][:]
            # print(lmList)
            
            # if len(lmList) != 0:

            #     header = [
            #                 'masked'+ str(i).zfill(3) + '.png: ',
            #                 '[' + str(lmList[0][1])  +', '+ str(int((lmList[11][1]+lmList[12][1])/2))    +', '+ str(lmList[12][1]) +', ' 
            #                     + str(lmList[14][1]) +', '+ str(lmList[16][1]) +', '+ str(lmList[11][1]) +', '+ str(lmList[13][1]) +', ' 
            #                     + str(lmList[15][1]) +', '+ str(lmList[24][1]) +', '+ str(lmList[26][1]) +', '+ str(lmList[28][1]) +', '
            #                     + str(lmList[23][1]) +', '+ str(lmList[25][1]) +', '+ str(lmList[27][1]) +', '+ str(lmList[2][1])  +', '
            #                     + str(lmList[5][1])  +', '+ str(lmList[7][1])  +', '+ str(lmList[8][1])  + ']: ',
            #                 '[' + str(lmList[0][2])  +', '+ str(int((lmList[11][2]+lmList[12][2])/2))    +', '+ str(lmList[12][2]) +', ' 
            #                     + str(lmList[14][2]) +', '+ str(lmList[16][2]) +', '+ str(lmList[11][2]) +', '+ str(lmList[13][2]) +', ' 
            #                     + str(lmList[15][2]) +', '+ str(lmList[24][2]) +', '+ str(lmList[26][2]) +', '+ str(lmList[28][2]) +', '
            #                     + str(lmList[23][2]) +', '+ str(lmList[25][2]) +', '+ str(lmList[27][2]) +', '+ str(lmList[2][2])  +', '
            #                     + str(lmList[5][2])  +', '+ str(lmList[7][2])  +', '+ str(lmList[8][2])  + ']'  
            #                 ]

                # write the header
                # writer.writerow(header)
            '''
                # xlist = []
                # ylist = []
                # name = []
                # name.append("masked"+ str(i).zfill(3) + '.png')
                # # nose
                # xlist.append(lmList[0][1])
                # ylist.append(lmList[0][2])
                # #neck
                # xlist.append((lmList[11][1]+lmList[12][1])/2)
                # ylist.append((lmList[11][2]+lmList[12][2])/2)
                # #Rsho
                # xlist.append(lmList[12][1])
                # ylist.append(lmList[12][2])
                # #Relb
                # xlist.append(lmList[14][1])
                # ylist.append(lmList[14][2])
                # #Rwri
                # xlist.append(lmList[16][1])
                # ylist.append(lmList[16][2])
                # #Lsho
                # xlist.append(lmList[11][1])
                # ylist.append(lmList[11][2])
                # #Lelb
                # xlist.append(lmList[13][1])
                # ylist.append(lmList[13][2])
                # #Lwri
                # xlist.append(lmList[15][1])
                # ylist.append(lmList[15][2])
                # #Rhip
                # xlist.append(lmList[24][1])
                # ylist.append(lmList[24][2])
                # #Rkne
                # xlist.append(lmList[26][1])
                # ylist.append(lmList[26][2])
                # #Rank
                # xlist.append(lmList[28][1])
                # ylist.append(lmList[28][2])
                # #Lhip
                # xlist.append(lmList[23][1])
                # ylist.append(lmList[23][2])
                # #Lkne
                # xlist.append(lmList[25][1])
                # ylist.append(lmList[25][2])
                # #Lank
                # xlist.append(lmList[27][1])
                # ylist.append(lmList[27][2])
                # #Leye
                # xlist.append(lmList[2][1])
                # ylist.append(lmList[2][2])
                # #Reye
                # xlist.append(lmList[5][1])
                # ylist.append(lmList[5][2])
                # #Lear
                # xlist.append(lmList[7][1])
                # ylist.append(lmList[7][2])
                # #Rear
                # xlist.append(lmList[8][1])
                # ylist.append(lmList[8][2])
            '''





                # print(lmList[14])
                # a = [0,11,12,14,16,13,15,24,26,28,23,25,27,2,5,7,8]
                # for i in a:
                #     cv2.circle(img, (lmList[i][1], lmList[i][2]), 5, (255, 0, 255), cv2.FILLED)
            # cTime = time.time()
            # fps = 1 / (cTime - pTime)
            # pTime = cTime
            # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
            #             (255, 0, 0), 3)
            cv2.imshow("Image", img)
            cv2.imshow("annotated_image", annotated_image)
            # cv2.imshow("masked_image", masked_image)
            # cv2.imshow("masked_image_", masked_image_)
            cv2.imshow("masked_image_eroded", masked_image_eroded)
            cv2.imshow("masked_image_black", masked_image_black)
            cv2.imshow("cropped image", cropped_image)


            # cv2.imwrite('./data/frames_masked/masked' + str(i).zfill(3) + '.png', masked_image_eroded)
            # cv2.imwrite('./data/frames_with_pose/annotated' + str(i).zfill(3) + '.png', annotated_image)
            # cv2.imwrite('./data/frames_all_pose/annotated' + str(i).zfill(3) + '.png', img)
            # cv2.imwrite('./data/frames_black_white/mask' + str(i).zfill(3) + '.png', masked_image_black)
            cv2.imwrite('./data/frames_cropped/cropped' + str(i).zfill(3) + '.png', cropped_image)
            i+=1
            cv2.waitKey(1)

if __name__ == "__main__":
    main()