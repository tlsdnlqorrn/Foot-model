# import cv2
# import numpy as np
#
# if __name__ == '__main__':
#
#
#     img = cv2.imread("image.jpg")
#
#     lower = np.array([0, 48, 80], dtype="uint8")
#     upper = np.array([20, 255, 255], dtype="uint8")
#
#     frame = cv2.flip(img, 1)
#     hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#     skinMask = cv2.inRange(hsv, lower, upper)
#
#     skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
#
#     ret, thresh = cv2.threshold(skinMask, 100, 255, cv2.THRESH_BINARY)
#     cv2.imshow("thresh",thresh)
#
#     contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = max(contours, key=lambda x: cv2.contourArea(x))
#     cv2.drawContours(frame, [contours], -1, (255, 255, 0), 3)
#
#     cv2.imshow("frame", frame)
#
#
#     cv2.waitKey(0)
import cv2
import numpy as np

#Open a simple image
img=cv2.imread("image.jpg")

#converting from gbr to hsv color space
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
HSV_mask = cv2.inRange(img_HSV, (0, 5, 0), (17,170,255))
HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

#converting from gbr to YCbCr color space
img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135))
YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

#merge skin detection (YCbCr and hsv)
global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
global_mask=cv2.medianBlur(global_mask,3)
global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))


HSV_result = cv2.bitwise_not(HSV_mask)
YCrCb_result = cv2.bitwise_not(YCrCb_mask)
global_result=cv2.bitwise_not(global_mask)


#show results
cv2.imshow("1_HSV.jpg",HSV_result)
cv2.imshow("2_YCbCr.jpg",YCrCb_result)
cv2.imshow("3_global_result.jpg",global_result)
#cv2.imshow("Image.jpg",img)
# cv2.imwrite("1_HSV.jpg",HSV_result)
# cv2.imwrite("2_YCbCr.jpg",YCrCb_result)
# cv2.imwrite("3_global_result.jpg",global_result)
cv2.waitKey(0)
cv2.destroyAllWindows()