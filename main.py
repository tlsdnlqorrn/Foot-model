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

# #Open a simple image
# img=cv2.imread("image.jpg")
#
# #converting from gbr to hsv color space
# img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# HSV_mask = cv2.inRange(img_HSV, (0, 5, 0), (17,170,255))
# HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
#
# #converting from gbr to YCbCr color space
# img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135))
# YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
#
# #merge skin detection (YCbCr and hsv)
# global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
# global_mask=cv2.medianBlur(global_mask,3)
# global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
#
#
# HSV_result = cv2.bitwise_not(HSV_mask)
# YCrCb_result = cv2.bitwise_not(YCrCb_mask)
# global_result=cv2.bitwise_not(global_mask)
#
#
# #show results
# cv2.imshow("1_HSV.jpg",HSV_result)
# cv2.imshow("2_YCbCr.jpg",YCrCb_result)
# cv2.imshow("3_global_result.jpg",global_result)
# #cv2.imshow("Image.jpg",img)
# # cv2.imwrite("1_HSV.jpg",HSV_result)
# # cv2.imwrite("2_YCbCr.jpg",YCrCb_result)
# # cv2.imwrite("3_global_result.jpg",global_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

if __name__ == '__main__':

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    frame = cv2.imread("image.jpg")
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    # frame = cv2.flip(frame, 1)
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
    # global_result = cv2.bitwise_not(global_mask)

    ret, thresh = cv2.threshold(global_mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours)==0:
    #     print("why")
    #     continue
    # else:
    cnt = contours[0]

    x, y, w, h = cv2.boundingRect(cnt)
    img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    global_result = cv2.bitwise_not(global_mask)
    white_up_point = []
    white_left_point = []
    if h > w and y < y + h:
        # 발가락이 위를 볼때
        img = cv2.line(img, (x, y + h), (x + w, y + h), (255, 0, 0), 2)
        for i in range(x, x + w):
            for j in range(y, y + h):
                print(global_result[i][j])
                if global_result[i][j] == 255:
                    white_up_point.append((i, j))
                    print(white_up_point)
                    break
    else:
        # 발가락이 왼쪽을 볼때
        img = cv2.line(img, (x + w, y), (x + w, y + h), (255, 0, 0), 2)
        for i in range(y, y + h):
            for j in range(global_result[x][i], global_result[x + w][i]):
                if global_result[i][j] == 255:
                    white_left_point.append((i, j))
                    # print(white_left_point)
                    break
    cv2.imshow("ycbcr", YCrCb_result)
    cv2.imshow("HSV", HSV_result)
    cv2.imshow("frame", global_mask)
    cv2.imshow("frame2", img)

cv2.waitKey(0)
cv2.destroyAllWindows()