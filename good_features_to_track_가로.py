# import cv2
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
#
# def skin_detector(frame):
#     img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     HSV_mask = cv2.inRange(img_HSV, (0, 5, 0), (17, 170, 255))
#     HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
#
#     # ycbcr 피부 영역 추출
#     img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
#     YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
#     YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
#
#     # and 연산으로 겹치는 것만 추출
#     global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
#     global_mask = cv2.medianBlur(global_mask, 3)
#     global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
#
#     return global_mask
#
# if __name__ == '__main__':
#
#     cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
#     lower = np.array([0, 48, 80], dtype="uint8")
#     upper = np.array([20, 255, 255], dtype="uint8")
#
#     while True:
#         _, frame = cap.read()
#         cv2.rectangle(frame,(180,60),(460,180),(0,0,255),5)
#         cv2.rectangle(frame,(180,300),(460,420),(0,0,255),5)
#         frame1=frame[60:180,180:460]
#         frame2=frame[300:420,180:460]
#         # cv2.line(frame, (380, 60), (380, 180), (0, 0, 255), 5)
#         # cv2.line(frame, (380, 300), (380, 420), (0, 0, 255), 5)
#
#         global_mask_1=skin_detector(frame1)
#         global_mask_2=skin_detector(frame2)
#
#         global_result_1 = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
#         global_result_2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
#
#         global_mask_1 = cv2.bitwise_not(global_mask_1)
#         global_mask_2 = cv2.bitwise_not(global_mask_2)
#
#         corners_1 = cv2.goodFeaturesToTrack(global_result_1, 100, 0.01, 5, blockSize=5, useHarrisDetector=False)
#         corners_2 = cv2.goodFeaturesToTrack(global_result_2, 100, 0.01, 5, blockSize=5, useHarrisDetector=False)
#
#         center_1=[]
#         center_2=[]
#         count_1=0
#         count_2=0
#         threshold=5
#         for i in corners_1:
#             center_1.append(tuple(i[0]))
#         center_1.sort(key=lambda x: x[0],reverse=True)
#
#         for i in range(0,len(center_1)):
#             if global_mask_1[int(center_1[i][1])][int(center_1[i][0])]==0 and center_1[i][0]+180>370:
#                 if count_1<threshold:
#                     count_1+=1
#                     print("1",center_1[i])
#                     cv2.line(frame, (int(center_1[i][0]+180),int(center_1[i][1])+60),(int(center_1[i][0]+180),int(center_1[i][1]+60)),(0, 0, 255), 10)
#                 else:
#                     break
#
#         for i in corners_2:
#             center_2.append(tuple(i[0]))
#         center_2.sort(key=lambda x: x[0],reverse=False)
#
#         for i in range(0,len(center_2)):
#             if global_mask_2[int(center_2[i][1])][int(center_2[i][0])]==0 and center_2[i][0]+180>370:
#                 if count_2 < threshold:
#                     print("2",center_2[i])
#                     count_2+=1
#                     cv2.line(frame, (int(center_2[i][0]+180),int(center_2[i][1]+300)),(int(center_2[i][0]+180),int(center_2[i][1]+300)),(0, 0, 255), 10)
#                 else:
#                     break
#
#         cv2.imshow("dst", frame)
#         cv2.imshow("global mask 1",global_mask_1)
#         cv2.imshow("global mask 2",global_mask_2)
#
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def skin_detector(frame):
    img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    HSV_mask = cv2.inRange(img_HSV, (0, 5, 0), (17, 170, 255))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # ycbcr 피부 영역 추출
    img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # and 연산으로 겹치는 것만 추출
    global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    return global_mask

if __name__ == '__main__':

    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    while True:
        _, frame = cap.read()
        cv2.rectangle(frame,(100,120),(600,360),(0,0,255),5)
        frame1=frame[120:360,100:600]

        global_mask_1=skin_detector(frame1)

        global_result_1 = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)

        global_mask_1 = cv2.bitwise_not(global_mask_1)

        corners_1 = cv2.goodFeaturesToTrack(global_result_1, 100, 0.01, 5, blockSize=5, useHarrisDetector=False)

        center_1=[]
        count_1=0
        threshold=5
        for i in corners_1:
            center_1.append(tuple(i[0]))
        center_1.sort(key=lambda x: x[0],reverse=True)

        for i in range(0,len(center_1)):
            if global_mask_1[int(center_1[i][1])][int(center_1[i][0])]==0 and center_1[i][0]+100>450:
                if count_1 < threshold:
                    count_1 += 1
                    cv2.line(frame, (int(center_1[i][0] + 100), int(center_1[i][1]) + 120),(int(center_1[i][0] + 100), int(center_1[i][1] + 120)), (0, 0, 255), 10)
                else:
                    break

        cv2.imshow("dst", frame)
        cv2.imshow("global mask 1",global_mask_1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cv2.waitKey(0)
cv2.destroyAllWindows()