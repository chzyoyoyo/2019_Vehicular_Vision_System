import cv2 as cv
from glob import glob


inputs = sorted((glob('./HW1_dataset/input/in*')))

back_sub = cv.createBackgroundSubtractorMOG2(history=len(inputs), varThreshold=180, detectShadows=False)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

i = 0

for i, img_path in enumerate(inputs):
    i += 1
    frame = cv.imread(img_path)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV)[:, :, 2]

    lr = -1 if i < 200 else 0
    fg_mask = back_sub.apply(frame, learningRate=lr)

    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)

    fg_mask = cv.dilate(fg_mask, kernel, iterations=30)
    fg_mask = cv.erode(fg_mask, kernel, iterations=30)

    cv.imwrite('./output/%.6d.png' % i, fg_mask)
