import os
import cv2
import numpy as np

if __name__ == '__main__':
    # raw data
    fuse_y_path = "D:/a0/0aaa/img/gray/"
    vi_path     = "D:/a0/0aaa/img/vi/"
    # building save path
    rgb_fuse_path = "D:/a0/0aaa/img/rgb/"
    if not os.path.exists(rgb_fuse_path):
        os.makedirs(rgb_fuse_path)
    # get raw image list
    y_file_list  = sorted(os.listdir(fuse_y_path))
    vi_file_list = sorted(os.listdir(vi_path))
    
    imgsize = (640, 480)

    for idx, (y_filename, vi_filename) in enumerate(zip(y_file_list, vi_file_list)):

        y_filepath  = os.path.join(fuse_y_path, y_filename)
        vi_filepath = os.path.join(vi_path, vi_filename)
        # read Y_fused image & visible image
        fuse_y  = cv2.imread(y_filepath,  flags=cv2.IMREAD_GRAYSCALE)
        img_vi  = cv2.imread(vi_filepath, flags=cv2.IMREAD_COLOR)
        
        img_vi = cv2.resize(img_vi, imgsize)
        # get cb and cr channels of the visible image
        vi_ycbcr = cv2.cvtColor(img_vi, cv2.COLOR_BGR2YCrCb)
        vi_y  = vi_ycbcr[:, :, 0]
        vi_cb = vi_ycbcr[:, :, 1]
        vi_cr = vi_ycbcr[:, :, 2]
        # get BGR-fused image
        fused_ycbcr = np.stack([fuse_y, vi_cb, vi_cr], axis=2).astype(np.uint8)
        fused_bgr = cv2.cvtColor(fused_ycbcr, cv2.COLOR_YCrCb2BGR)
        # save RGB-fused image
        bgr_fuse_save_name = os.path.join(rgb_fuse_path, y_filename)
        print(bgr_fuse_save_name)
        cv2.imwrite(bgr_fuse_save_name, fused_bgr)
