import cv2
import numpy as np

if __name__ == '__main__':


    cap = cv2.VideoCapture(1)

    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # skinMask = cv2.inRange(hsv, lower, upper)
        #
        # skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        #
        # ret, thresh = cv2.threshold(skinMask, 100, 255, cv2.THRESH_BINARY)
        # cv2.imshow("thresh",thresh)
        #
        # contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours = max(contours, key=lambda x: cv2.contourArea(x))
        # cv2.drawContours(frame, [contours], -1, (255, 255, 0), 3)
        #
        # cv2.imshow("frame", frame)
        img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        HSV_mask = cv2.inRange(img_HSV, (0, 5, 0), (17, 170, 255))
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # converting from gbr to YCbCr color space
        img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # merge skin detection (YCbCr and hsv)
        global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
        global_mask = cv2.medianBlur(global_mask, 3)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        HSV_result = cv2.bitwise_not(HSV_mask)
        YCrCb_result = cv2.bitwise_not(YCrCb_mask)
        global_result = cv2.bitwise_not(global_mask)
        cv2.imshow("frame", global_result)
        cv2.imshow("frame2",YCrCb_result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()