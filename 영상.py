import cv2
import numpy as np

if __name__ == '__main__':


    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    while True:
        _, frame = cap.read()
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
        
        #hsv 피부 영역 추출
        img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        HSV_mask = cv2.inRange(img_HSV, (0, 5, 0), (17, 170, 255))
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        #ycbcr 피부 영역 추출
        img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        #and 연산으로 겹치는 것만 추출
        global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
        global_mask = cv2.medianBlur(global_mask, 3)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        HSV_result = cv2.bitwise_not(HSV_mask)
        YCrCb_result = cv2.bitwise_not(YCrCb_mask)
        # global_result = cv2.bitwise_not(global_mask)

        ret, thresh = cv2.threshold(global_mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==0:
            continue
        else:
            cnt = contours[0]

        x, y, w, h = cv2.boundingRect(cnt)
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)



        # global_mask=global_mask[y:y+h,x:x+w]
        global_result = cv2.bitwise_not(global_mask)
        black_up_distance=[]
        black_left_distance=[]
        black_up_points=[]

        img = cv2.line(img, (x, y + h), (x + w, y + h), (255, 0, 0), 2)
        global_result = cv2.line(global_result, (x, y + h), (x + w, y + h), (0, 0, 0), 2)
        line, _ = cv2.findContours(global_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, line, -1, (0, 255, 0), 1)
        if h>w and w>30:
            #발가락이 위를 볼때
            for i in range(x, x + w):
                for j in range(y+h,y,-1):
                            img = cv2.line(img, (x, y + h), (x, y + h), (0, 255, 0), 10)
                            img = cv2.line(img, (x + w, y + h), (x + w, y + h), (0, 255, 0), 10)
                            img = cv2.line(img, (x, y), (x, y), (255, 255, 255), 10)
                            img = cv2.line(img, (x + w, y), (x + w, y), (0, 0, 0), 10)
                            if global_result[i][j]==0:
                                img=cv2.line(img,(i,j),(i,j),(0,0,0),2)
                                black_up_distance.append(y+h-j)
                                black_up_points.append((i,j))
                                break
                # print(len(black_up_distance)==w)

            # else:
            #     #발가락이 왼쪽을 볼때
            #     img = cv2.line(img, (x+w, y), (x+w, y + h), (255, 0, 0), 2)
            #     for i in range(y,y+h):
            #         for j in range(global_result[x][i],global_result[x+w][i]):
            #             if global_result[i][j]==255:
            #                 black_left_distance.append((i,j))
            #                 # print(black_left_distance)
            #                 break
            diff=[black_up_distance[i]-black_up_distance[i-3] for i in range(3,len(black_up_distance))]
            zerodistance=[black_up_points[i] for i in range(1,len(diff)) if (diff[i-1]*diff[i])<0]
            for i in range(0,len(zerodistance)):
                cv2.line(img,zerodistance[i],zerodistance[i],(0,255,255),5)

            cv2.imshow("frame2", img)

        cv2.imshow("ycbcr",YCrCb_result)
        cv2.imshow("HSV",HSV_result)
        cv2.imshow("frame", global_result)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

#     for i in range(x,x+w):
        #         for j in range(y+h,y,-1):
        #             img = cv2.line(img, (x, y + h), (x, y + h), (0, 255, 0), 10)
        #             img = cv2.line(img, (x + w, y + h), (x + w, y + h), (0, 255, 0), 10)
        #             img = cv2.line(img, (x, y), (x, y), (255, 255, 255), 10)
        #             img = cv2.line(img, (x + w, y), (x + w, y), (0, 0, 0), 10)
        #             if global_result[i][j]==0:
        #                 img=cv2.line(img,(i,j),(i,j),(0,0,0),2)
        #                 black_up_distance.append(y+h-j)
        #                 black_up_points.append((i,j))
        #                 break
        #     # print(len(black_up_distance)==w)
        #
        # # else:
        # #     #발가락이 왼쪽을 볼때
        # #     img = cv2.line(img, (x+w, y), (x+w, y + h), (255, 0, 0), 2)
        # #     for i in range(y,y+h):
        # #         for j in range(global_result[x][i],global_result[x+w][i]):
        # #             if global_result[i][j]==255:
        # #                 black_left_distance.append((i,j))
        # #                 # print(black_left_distance)
        # #                 break
        #     diff=[black_up_distance[i]-black_up_distance[i-3] for i in range(3,len(black_up_distance))]
        #     zerodistance=[black_up_points[i] for i in range(1,len(diff)) if (diff[i-1]*diff[i])<0]
        #     for i in range(0,len(zerodistance)):
        #         cv2.line(img,zerodistance[i],zerodistance[i],(0,255,255),5)

# for i in range(len(line[0])):
#     black_up_distance.append(line[0][i])
# black_up_distance = np.array(black_up_distance)
# diff = [black_up_distance[i][0][1] - black_up_distance[i - 3][0][1] for i in range(3, len(black_up_distance[0]))]
# zerodistance = [black_up_points[i] for i in range(1, len(diff)) if (diff[i - 1] * diff[i]) < 0]
# for i in range(0, len(zerodistance)):
#     cv2.line(img, zerodistance[i], zerodistance[i], (0, 0, 0), 5)