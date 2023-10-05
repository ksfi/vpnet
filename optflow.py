import cv2
import os
import numpy as np

DSIZE = (100, 100)

def computeflow(frame1, frame2):
    frame1 = frame1[100:650]
    frame1 = cv2.resize(frame1, (0,0), fx = 0.5, fy=0.5)
    frame2 = frame2[100:650]
    frame2 = cv2.resize(frame2, (0,0), fx = 0.5, fy=0.5)
    flow = np.zeros_like(frame1)
    prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow_data = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.4, 1, 12, 2, 8, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow_data[...,0], flow_data[...,1])
    flow[...,1] = 255
    flow[...,0] = ang*180/np.pi/2
    flow[...,2] = (mag *15).astype(int)
    return flow

def optflow(video_path):
    j = 0
    for k in range(1, 5):
        video_file = video_path
        optflow_dir = f"_eval/{k}"
        try:
            os.mkdir(optflow_dir)
        except: pass
        vid = cv2.VideoCapture(video_file)
        ret, prev = vid.read()
        i = 0
        while ret:
            ret, nxt  = vid.read()
            if nxt is None: break
            i+=1
            j+=1
            print(j)
            flow = computeflow(prev,nxt)
            flow = cv2.resize(flow, DSIZE, interpolation = cv2.INTER_AREA)
            cv2.imwrite(optflow_dir + '/' + str(j) + ".png", flow/127.5-1.)
            print(f"frame: {i} video: {k}")
            prev = nxt

if __name__ == "__main__":
    optflow()