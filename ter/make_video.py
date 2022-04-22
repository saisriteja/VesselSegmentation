import cv2
import numpy as np
import os 

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

width = 780

height = 540

video = cv2.VideoWriter('opticalflow.mp4', fourcc, 1, (width, height))

no_of_files = len(os.listdir('D:\\projects\\pro\\VesselSegmentation\\ter\\DenseOutput')) // 2


for j in range(no_of_files):
    # D:\projects\pro\VesselSegmentation\ter\DenseOutput\original_0.png
    # print(f'/DenseOutput/original_{str(j)}.png')
    img1 = cv2.imread(f'D:\\projects\\pro\\VesselSegmentation\\ter\\DenseOutput\\original_{j}.png')
    img2 = cv2.imread(f'D:\\projects\\pro\\VesselSegmentation\\ter\\DenseOutput\\rgb_{j}.png')

    vis = np.concatenate((img1, img2), axis=1)

    vis = cv2.resize(vis, (780, 540), interpolation = cv2.INTER_NEAREST)

    video.write(vis)

cv2.destroyAllWindows()
video.release()







