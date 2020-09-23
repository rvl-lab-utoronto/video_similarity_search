import cv2
import numpy as np
cap = cv2.VideoCapture("/media/diskstation/datasets/UCF101/videos/TaiChi/v_TaiChi_g10_c01.avi")
ret, frame1 = cap.read()

h = frame1.shape[0]
w = frame1.shape[1]
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (w,  h))

while(1):
    ret, frame2 = cap.read()
    if frame2 is None:
        break
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    to_dislay = np.hstack((frame2, rgb))
    cv2.imshow('frame2',to_dislay)
    out.write(to_dislay)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

cap.release()
out.release()
cv2.destroyAllWindows()
