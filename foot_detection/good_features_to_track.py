import cv2
import numpy as np

if __name__ == '__main__':

    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    while True:
        _, frame = cap.read()
        # hsv 피부 영역 추출
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

        HSV_result = cv2.bitwise_not(HSV_mask)
        YCrCb_result = cv2.bitwise_not(YCrCb_mask)
        global_result = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

        global_mask = cv2.bitwise_not(global_mask)
        corners = cv2.goodFeaturesToTrack(global_result, 100, 0.01, 5, blockSize=3, useHarrisDetector=True, k=0.03)

        for i in corners:
            center=tuple(i[0])
            print(center)
            if global_mask[int(center[1])][int(center[0])]==0:
                cv2.circle(global_result, (int(center[0]),int(center[1])), 3, (0, 0, 255), 2)

        cv2.imshow("dst", global_result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cv2.waitKey(0)
cv2.destroyAllWindows()