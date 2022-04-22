import cv2
import numpy as np
cap = cv2.VideoCapture("terrorist.mp4")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255


c = 0

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    rgb = cv2.resize(rgb, (780, 540),interpolation = cv2.INTER_NEAREST)
    frame2 = cv2.resize(frame2, (780, 540),interpolation = cv2.INTER_NEAREST)

    # cv2.imshow('frame2',rgb)
    # cv2.imshow('general', frame2)

    cv2.imwrite(f'DenseOutput/original_{c}.png',frame2)
    cv2.imwrite(f'DenseOutput/rgb_{c}.png',rgb)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

    c = c + 1

cap.release()
cv2.destroyAllWindows()