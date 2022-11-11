import cv2
import numpy as np
import time

from modules.rppg import rPPG
from sklearn.neighbors import KNeighborsClassifier


def skin_detector(frame):
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

    return global_mask


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

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    path = 'C:/Users/shinwi/Desktop/FootModel/mine.mp4'
    #pcap = cv2.VideoCapture(path)


    rppg_switch = False

    while True:
        _, frame = cap.read()

        global_mask = skin_detector(frame)

        ret, thresh = cv2.threshold(global_mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        global_result = cv2.bitwise_not(global_mask)
        black_up_distance = []
        black_left_distance = []
        black_up_points = []

        img = cv2.line(img, (x, y + h), (x + w, y + h), (255, 0, 0), 2)

        # img = cv2.line(img, (x + round(w/3), y + round(h/7*5)), (x + round(w/3), y + h), (0, 0, 255), 2)
        # img = cv2.line(img, (x + round(w/2), y + round(h/7*5)), (x + round(w/2), y + h), (0, 0, 255), 2)
        # img = cv2.line(img, (x + round(w/3*2), y + round(h/7*5)), (x + round(w/3*2), y + h), (0, 0, 255), 2)
        # img = cv2.line(img, (x + round(w/6*5), y + round(h/7*5)), (x + round(w/6*5), y + h), (0, 0, 255), 2)
        # img = cv2.line(img, (x, y + round(h/7*5)), (x + w, y + round(h/7*5)), (0, 0, 255), 2)

        ppg_screen1 = np.zeros((300, 500, 3), np.uint8)
        ppg_screen2 = np.zeros((300, 500, 3), np.uint8)
        ppg_screen3 = np.zeros((300, 500, 3), np.uint8)
        ppg_screen4 = np.zeros((300, 500, 3), np.uint8)
        ppg_screen5 = np.zeros((300, 500, 3), np.uint8)
        ppg_screen_all = np.zeros((300, 500, 3), np.uint8)

        img = cv2.rectangle(img, (x, y + round(h/7*5)), (x + round(w/3), y + h), (0, 0, 255), 2)
        img = cv2.rectangle(img, (x + round(w/3), y + round(h/7*5)), (x + round(w/2), y + h), (0, 0, 255), 2)
        img = cv2.rectangle(img, (x + round(w/2), y + round(h/7*5)), (x + round(w/3*2), y + h), (0, 0, 255), 2)
        img = cv2.rectangle(img, (x + round(w/3*2), y + round(h/7*5)), (x + round(w/6*5), y + h), (0, 0, 255), 2)
        img = cv2.rectangle(img, (x + round(w/6*5), y + round(h/7*5)), (x + w, y + h), (0, 0, 255), 2)

        if h > w > 30:
            # 발 가장자리 라인 그리기
            line, _ = cv2.findContours(global_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, line, -1, (0, 255, 0), 2)

            for i in range(x, x + w):
                for j in range(y + h, y, -1):
                    img = cv2.line(img, (x, y + h), (x, y + h), (0, 0, 0), 10)  # 사각형 왼쪽 아래 꼭짓점
                    #img = cv2.putText(img, str((x, y + h)), (x, y + h), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
                    img = cv2.line(img, (x + w, y + h), (x + w, y + h), (0, 0, 0), 10)  # 사각형 오른쪽 아래 꼭짓점
                    img = cv2.line(img, (x, y), (x, y), (0, 0, 0), 10)  # 사각형 왼쪽 위 꼭짓점
                    img = cv2.line(img, (x + w, y), (x + w, y), (0, 0, 0), 10)  # 사각형 오른쪽 위 꼭짓점

                    # 검은 영역일 때 = 발 영역일 때
                    if global_result[j][i] == 0:
                        cv2.line(img, (i, j), (i, j), (255, 255, 0), 3)
                        black_up_distance.append(y + h - j)
                        black_up_points.append((i, j))
                        break

        if rppg_switch:
            ppg1, bpm1, light_val1, cr_va1l, cb_val1 = rppg_1.process(frame[y + int(round(h/7*5)):y + h, x: x + int(round(w/3))])
            ppg2, bpm2, light_val2, cr_val2, cb_val2 = rppg_2.process(frame[y + int(round(h/7*5)):y + h, x + int(round(w/3)):x + int(round(w/2))])
            ppg3, bpm3, light_val3, cr_val3, cb_val3 = rppg_3.process(frame[y + int(round(h/7*5)):y + h, x + int(round(w/2)):x + int(round(w/3*2))])
            ppg4, bpm4, light_val4, cr_val4, cb_val4 = rppg_4.process(frame[y + int(round(h/7*5)):y + h, x + int(round(w/3*2)):x + int(round(w/6*5))])
            ppg5, bpm5, light_val5, cr_val5, cb_val5 = rppg_5.process(frame[y + int(round(h/7*5)):y + h, x + int(round(w/6*5)):x+w])
            ppg_all, bpm_all, light_val_all, cr_val_all, cb_val_all = rppg_all.process(frame[y:h+h, x:x+w])

            # cv2.rectangle(frame, (toe_area[0][0] - rect_size, toe_area[0][1] - rect_size), (toe_area[0][0] + rect_size, toe_area[0][1] + rect_size), (0, 0, 255), 3)
            # cv2.rectangle(frame, (toe_area[1][0] - rect_size, toe_area[1][1] - rect_size), (toe_area[1][0] + rect_size, toe_area[1][1] + rect_size), (0, 0, 255), 3)
            # cv2.rectangle(frame, (toe_area[2][0] - rect_size, toe_area[2][1] - rect_size), (toe_area[2][0] + rect_size, toe_area[2][1] + rect_size), (0, 0, 255), 3)
            # cv2.rectangle(frame, (toe_area[3][0] - rect_size, toe_area[3][1] - rect_size), (toe_area[3][0] + rect_size, toe_area[3][1] + rect_size), (0, 0, 255), 3)
            # cv2.rectangle(frame, (toe_area[4][0] - rect_size, toe_area[4][1] - rect_size), (toe_area[4][0] + rect_size, toe_area[4][1] + rect_size), (0, 0, 255), 3)

            draw_signal(ppg1, ppg_screen1)
            draw_signal(ppg2, ppg_screen2)
            draw_signal(ppg3, ppg_screen3)
            draw_signal(ppg4, ppg_screen4)
            draw_signal(ppg5, ppg_screen5)
            draw_signal(ppg_all, ppg_screen_all)

            cv2.putText(ppg_screen1, '%03d' % round(bpm1), (ppg_screen1.shape[1] - 230, ppg_screen1.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            cv2.putText(ppg_screen2, '%03d' % round(bpm2), (ppg_screen2.shape[1] - 230, ppg_screen2.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            cv2.putText(ppg_screen3, '%03d' % round(bpm3), (ppg_screen3.shape[1] - 230, ppg_screen3.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            cv2.putText(ppg_screen4, '%03d' % round(bpm4), (ppg_screen4.shape[1] - 230, ppg_screen4.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            cv2.putText(ppg_screen5, '%03d' % round(bpm5), (ppg_screen5.shape[1] - 230, ppg_screen5.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
            cv2.putText(ppg_screen_all, '%03d' % round(bpm_all), (ppg_screen_all.shape[1] - 230, ppg_screen_all.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)


        cv2.imshow("1", ppg_screen1)
        cv2.imshow("2", ppg_screen2)
        cv2.imshow("3", ppg_screen3)
        cv2.imshow("4", ppg_screen4)
        cv2.imshow("5", ppg_screen5)
        cv2.imshow("all", ppg_screen_all)


        cv2.imshow("frame", global_result)
        cv2.imshow("frame", img)

        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == ord("q"):
            break
        elif pressedKey == ord("p"):
            cv2.waitKey(-1)
        elif pressedKey == ord("s"):
            rppg_1 = rPPG()
            rppg_2 = rPPG()
            rppg_3 = rPPG()
            rppg_4 = rPPG()
            rppg_5 = rPPG()
            rppg_all = rPPG()
            rppg_switch=True

cv2.waitKey(0)
cv2.destroyAllWindows()
