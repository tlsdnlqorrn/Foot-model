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
import time
from modules.rppg import rPPG

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
    global_mask = cv2.dilate(global_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)

    return global_mask


def get_median(data):
    data = sorted(data)
    centerIndex = len(data) // 2
    return (int((data[centerIndex][0] + data[centerIndex - 1][0]) / 2),int((data[centerIndex][1] + data[centerIndex - 1][1]) / 2))

def toe_detect(elapsed_time,frame,toe_area,switch,rppg_switch):
    time_limit=5
    if int(elapsed_time) < time_limit:
        elapsed_time = time.time() - start_time
        print(elapsed_time)
        center_1 = []
        count_1 = 0
        threshold = 10
        points = []

        for i in corners_1:
            center_1.append(tuple(i[0]))
        center_1.sort(key=lambda x: x[0], reverse=False)

        for i in range(0, len(center_1)):
            if global_mask_1[int(center_1[i][1])][int(center_1[i][0])] == 0 and center_1[i][0] + 100 > 450:
                if count_1 < threshold:
                    count_1 += 1
                    points.append((int(center_1[i][0] + 100), int(center_1[i][1] + 120)))
                    # cv2.line(frame, (int(center_1[i][0] + 100), int(center_1[i][1]) + 120),(int(center_1[i][0] + 100), int(center_1[i][1] + 120)), (0, 0,255), 10)
                else:
                    break
        diff_x = [points[i][0] - points[i - 1][0] for i in range(1, len(points))]
        zerodistance = [points[i - 1] for i in range(1, len(diff_x)) if (diff_x[i - 1] * diff_x[i]) <= 0]
        zerodistance.sort(key=lambda x: x[1], reverse=True)

        if len(zerodistance) > 1 and abs(zerodistance[0][1] - zerodistance[-1][1]) > 120:
            toe_area_1.append((zerodistance[0][0], int((zerodistance[0][1] - zerodistance[-1][1])) + zerodistance[-1][1] - 50))
            toe_area_2.append((zerodistance[0][0],int((zerodistance[0][1] - zerodistance[-1][1]) / 5 * 4) + zerodistance[-1][1] - 50))
            toe_area_3.append((zerodistance[0][0],int((zerodistance[0][1] - zerodistance[-1][1]) / 5 * 3) + zerodistance[-1][1] - 40))
            toe_area_4.append((zerodistance[0][0],int((zerodistance[0][1] - zerodistance[-1][1]) / 5 * 2) + zerodistance[-1][1] - 30))
            toe_area_5.append((zerodistance[0][0], int((zerodistance[0][1] - zerodistance[-1][1]) / 5) + zerodistance[-1][1] - 20))

            cv2.line(frame, (zerodistance[0][0], zerodistance[0][1]), (zerodistance[-1][0], zerodistance[-1][1]),(0, 0, 255), 3)
            cv2.line(frame, (zerodistance[0][0], int((zerodistance[0][1] - zerodistance[-1][1]) / 5) + zerodistance[-1][1] - 20), (zerodistance[-1][0],int((zerodistance[0][1] - zerodistance[-1][1]) / 5) + zerodistance[-1][1] - 20), (0, 255, 0), 10)
            cv2.line(frame, (zerodistance[0][0], int((zerodistance[0][1] - zerodistance[-1][1]) / 5 * 2) + zerodistance[-1][1] - 30), (zerodistance[-1][0],int((zerodistance[0][1] - zerodistance[-1][1]) / 5) * 2 + zerodistance[-1][1] - 30), (0, 255, 0),10)
            cv2.line(frame, (zerodistance[0][0], int((zerodistance[0][1] - zerodistance[-1][1]) / 5 * 3) + zerodistance[-1][1] - 40), (zerodistance[-1][0],int((zerodistance[0][1] - zerodistance[-1][1]) / 5 * 3) + zerodistance[-1][1] - 40), (0, 255, 0),10)
            cv2.line(frame, (zerodistance[0][0], int((zerodistance[0][1] - zerodistance[-1][1]) / 5 * 4) + zerodistance[-1][1] - 50), (zerodistance[-1][0],int((zerodistance[0][1] - zerodistance[-1][1]) / 5 * 4) + zerodistance[-1][1] - 50), (0, 255, 0),10)
            cv2.line(frame, (zerodistance[0][0], int((zerodistance[0][1] - zerodistance[-1][1])) + zerodistance[-1][1] - 50),(zerodistance[-1][0], int((zerodistance[0][1] - zerodistance[-1][1])) + zerodistance[-1][1] - 50),(0, 255, 0), 10)
            toe_area = [get_median(toe_area_1), get_median(toe_area_2), get_median(toe_area_3), get_median(toe_area_4),get_median(toe_area_5)]
    if int(elapsed_time)==time_limit:
        switch=False
        rppg_switch=True
    return toe_area,switch,rppg_switch

def draw_signal(signal, frame):
    height, width = frame.shape[:2]

    # Signal preprocessing
    np_signal = np.array(signal)
    diff_val = np_signal.max() - np_signal.min()
    np_signal = np_signal if diff_val == 0 else (np_signal - np_signal.min()) / diff_val

    # Draw signal
    width_offset = width / np_signal.shape[0]
    for i in range(np_signal.shape[0] - 1):
        sx = i * width_offset
        sy = height - (np_signal[i] * height)
        ex = (i + 1) * width_offset
        ey = height - (np_signal[(i + 1)] * height)
        cv2.line(frame, (int(sx), int(sy)), (int(ex), int(ey)), (0, 255, 0), 3)

if __name__ == '__main__':

    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

    toe_area_1=[]
    toe_area_2=[]
    toe_area_3=[]
    toe_area_4=[]
    toe_area_5=[]
    toe_area=[(0,0),(0,0),(0,0),(0,0),(0,0)]
    switch=False

    rppg_switch=False
    rect_size=10

    while True:
        _, frame = cap.read()
        cv2.rectangle(frame,(100,120),(600,360),(0,0,255),5)
        frame1=frame[120:360,100:600]

        ppg_screen1 = np.zeros((300, 500, 3), np.uint8)
        ppg_screen2 = np.zeros((300, 500, 3), np.uint8)
        ppg_screen3 = np.zeros((300, 500, 3), np.uint8)
        ppg_screen4 = np.zeros((300, 500, 3), np.uint8)
        ppg_screen5 = np.zeros((300, 500, 3), np.uint8)

        global_mask_1=skin_detector(frame1)
        global_mask_1 = cv2.bitwise_not(global_mask_1)
        corners_1 = cv2.goodFeaturesToTrack(global_mask_1, 1000, 0.01, 5, blockSize=5, useHarrisDetector=False)

        if switch:
            elapsed_time = time.time() - start_time
            toe_area,switch,rppg_switch = toe_detect(elapsed_time, frame,toe_area,switch,rppg_switch)

        cv2.line(frame,(toe_area[0][0],toe_area[0][1]),(toe_area[0][0],toe_area[0][1]),(0,255,0),3)
        cv2.line(frame, (toe_area[1][0], toe_area[1][1]),(toe_area[1][0], toe_area[1][1]), (0, 255, 0), 3)
        cv2.line(frame, (toe_area[2][0], toe_area[2][1]),(toe_area[2][0], toe_area[2][1]), (0, 255, 0), 3)
        cv2.line(frame, (toe_area[3][0], toe_area[3][1]),(toe_area[3][0], toe_area[3][1]), (0, 255, 0), 3)
        cv2.line(frame, (toe_area[4][0], toe_area[4][1]),(toe_area[4][0], toe_area[4][1]), (0, 255, 0), 3)

        if rppg_switch:
            ppg1, bpm1, light_val1, cr_va1l, cb_val1=rppg_1.process(frame[toe_area[0][1]-rect_size:toe_area[0][1]+rect_size,toe_area[0][0]-rect_size:toe_area[0][0]+rect_size])
            ppg2, bpm2, light_val2, cr_val2, cb_val2=rppg_2.process(frame[toe_area[1][1] - rect_size:toe_area[1][1] + rect_size, toe_area[1][0] - rect_size:toe_area[1][0] + rect_size])
            ppg3, bpm3, light_val3, cr_val3, cb_val3=rppg_3.process(frame[toe_area[2][1] - rect_size:toe_area[2][1] + rect_size, toe_area[2][0] - rect_size:toe_area[2][0] + rect_size])
            ppg4, bpm4, light_val4, cr_val4, cb_val4=rppg_4.process(frame[toe_area[3][1] - rect_size:toe_area[3][1] + rect_size, toe_area[3][0] - rect_size:toe_area[3][0] + rect_size])
            ppg5, bpm5, light_val5, cr_val5, cb_val5=rppg_5.process(frame[toe_area[4][1] - rect_size:toe_area[4][1] + rect_size, toe_area[4][0] - rect_size:toe_area[4][0] + rect_size])

            cv2.rectangle(frame, (toe_area[0][0] - rect_size, toe_area[0][1] - rect_size), (toe_area[0][0] + rect_size, toe_area[0][1] + rect_size),(0, 0, 255), 3)
            cv2.rectangle(frame, (toe_area[1][0] - rect_size, toe_area[1][1] - rect_size), (toe_area[1][0] + rect_size, toe_area[1][1] + rect_size),(0, 0, 255), 3)
            cv2.rectangle(frame, (toe_area[2][0] - rect_size, toe_area[2][1] - rect_size), (toe_area[2][0] + rect_size, toe_area[2][1] + rect_size),(0, 0, 255), 3)
            cv2.rectangle(frame, (toe_area[3][0] - rect_size, toe_area[3][1] - rect_size), (toe_area[3][0] + rect_size, toe_area[3][1] + rect_size),(0, 0, 255), 3)
            cv2.rectangle(frame, (toe_area[4][0] - rect_size, toe_area[4][1] - rect_size), (toe_area[4][0] + rect_size, toe_area[4][1] + rect_size),(0, 0, 255), 3)

            draw_signal(ppg1,ppg_screen1)
            draw_signal(ppg2, ppg_screen2)
            draw_signal(ppg3, ppg_screen3)
            draw_signal(ppg4, ppg_screen4)
            draw_signal(ppg5, ppg_screen5)

            cv2.putText(ppg_screen1, '%03d' % round(bpm1), (ppg_screen1.shape[1] - 230, ppg_screen1.shape[0] - 50),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            cv2.putText(ppg_screen2, '%03d' % round(bpm2), (ppg_screen2.shape[1] - 230, ppg_screen2.shape[0] - 50),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            cv2.putText(ppg_screen3, '%03d' % round(bpm3), (ppg_screen3.shape[1] - 230, ppg_screen3.shape[0] - 50),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            cv2.putText(ppg_screen4, '%03d' % round(bpm4), (ppg_screen4.shape[1] - 230, ppg_screen4.shape[0] - 50),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            cv2.putText(ppg_screen5, '%03d' % round(bpm5), (ppg_screen5.shape[1] - 230, ppg_screen5.shape[0] - 50),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

        cv2.imshow("1",ppg_screen1)
        cv2.imshow("2",ppg_screen2)
        cv2.imshow("3",ppg_screen3)
        cv2.imshow("4",ppg_screen4)
        cv2.imshow("5",ppg_screen5)
        cv2.imshow("dst", frame)
        cv2.imshow("global mask 1",global_mask_1)
        key=cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key==ord("s"):
            start_time = time.time()
            rppg_1 = rPPG()
            rppg_2 = rPPG()
            rppg_3 = rPPG()
            rppg_4 = rPPG()
            rppg_5 = rPPG()
            toe_area = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
            rppg_switch=False
            switch=True

cv2.waitKey(0)
cv2.destroyAllWindows()