import cv2
import numpy as np

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
        cv2.rectangle(frame,(120,120),(240,360),(0,0,255),5)
        cv2.rectangle(frame,(400,120),(520,360),(0,0,255),5)
        frame1=frame[120:360,120:240]
        frame2=frame[120:360,400:520]

        global_mask_1=skin_detector(frame1)
        global_mask_2=skin_detector(frame2)

        global_result_1 = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
        global_result_2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        global_mask_1 = cv2.bitwise_not(global_mask_1)
        global_mask_2 = cv2.bitwise_not(global_mask_2)

        corners_1 = cv2.goodFeaturesToTrack(global_result_1, 1000, 0.01, 5, blockSize=5, useHarrisDetector=True, k=0.03)
        corners_2 = cv2.goodFeaturesToTrack(global_result_2, 1000, 0.01, 5, blockSize=5, useHarrisDetector=True, k=0.03)

        for i in corners_1:
            center=tuple(i[0])
            # print(center)
            if global_mask_1[int(center[1])][int(center[0])]==0 and center[1]+120>280:
                cv2.line(frame, (int(center[0]+120),int(center[1])+120),(int(center[0]+120),int(center[1]+120)),(0, 0, 255), 10)

        for i in corners_2:
            center=tuple(i[0])
            # print(center)
            if global_mask_2[int(center[1])][int(center[0])]==0 and center[1]+120>280:
                cv2.line(frame, (int(center[0]+400),int(center[1]+120)),(int(center[0]+400),int(center[1]+120)),(0, 0, 255), 10)

        cv2.imshow("dst", frame)
        cv2.imshow("global mask 1",global_mask_1)
        cv2.imshow("global mask 2",global_mask_2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cv2.waitKey(0)
cv2.destroyAllWindows()