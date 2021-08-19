import numpy as np
import cv2 as cv


slika = cv.imread('input.png')

slikaMedian = cv.medianBlur(slika, 11)


slikaHSV = cv.cvtColor(slikaMedian, cv.COLOR_BGR2HSV)
slikaHue = slikaHSV[:, :, 0]


slikaTh = cv.inRange(slikaHue, 0, 40)
cv.imwrite('range.png', slikaTh)


kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
slikaOpen = cv.morphologyEx(src=slikaTh, op=cv.MORPH_OPEN, kernel=kernel)



slikaOut = slika.copy()
cntCC, imgCC = cv.connectedComponents(slikaOpen, connectivity=4)


maxCnt = 0
maxBBox = None
for cc in range(1, cntCC):
    imgCurr = np.where(imgCC == cc, 255, 0).astype(np.uint8)
    x, y, w, h = cv.boundingRect(imgCurr)
    cnt = imgCurr.sum() / 255
    if cnt > maxCnt:
        maxCnt = cnt
        maxBBox = x, y, w, h
    cv.rectangle(slikaOut, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)


x, y, w, h = maxBBox
cv.rectangle(slikaOut, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)


cv.putText(slikaOut, text='CNT: ' + str(cntCC - 1), org=(5, 17), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
           color=(0, 0, 255), thickness=2)
cv.imshow("Output", slikaOut)
cv.imwrite('output.png', slikaOut)

cv.waitKey(0)
cv.destroyAllWindows()
