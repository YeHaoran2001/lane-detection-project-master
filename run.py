import numpy as np
import cv2
import utlis
import argparse
from preprocessing.Dehaze import dehaze
from preprocessing.Deshadow import process_image_file as deshadow
from preprocessing.Illumination import enhance

def main(opts):

    def pipeline(img):
        img_original = cv2.resize(img, (frameWidth, frameHeight))
        # STEP 1: data preprocessing
        if opts.dehaze:
            img = dehaze(img)
        if opts.deshadow:
            _, _, img = deshadow(img)
        if opts.illumination:
            img = enhance(img)
        # STEP 2: resize the image
        img = cv2.resize(img, (frameWidth, frameHeight), None)
        imgWarpPoints = img.copy()
        imgFinal = img.copy()
        imgCanny = img.copy()

        # STEP 3: applying Canny edge detection, dilating, thresholding
        imgThres, imgCanny, imgColor, imgDilate = utlis.thresholding(img, opts=opts)

        src = utlis.valTrackbars()

        # STEP 4: perspective warp
        imgWarp = utlis.perspective_warp(imgThres, dst_size=(frameWidth, frameHeight), src=src)
        imgWarpPoints = utlis.drawPoints(imgWarpPoints, src)

        # STEP 5: detect lanes based on sliding window 
        imgSliding, curves, lanes, ploty = utlis.sliding_window(imgWarp, draw_windows=True)
        imgFinal = utlis.draw_lanes(img, curves[0], curves[1], frameWidth, frameHeight, src=src)


        imgStacked = utlis.stackImages(0.7, ([img_original,img,imgWarpPoints],
                                                [imgDilate, imgColor, imgThres],
                                                [imgWarp,imgSliding,imgFinal],
                                                ))
        return imgStacked, imgFinal


    # initialize parameters
    path = opts.path

    frameWidth = opts.frameWidth
    frameHeight = opts.frameHeight
    intialTracbarVals = opts.intialTracbarVals

    utlis.initializeTrackbars(intialTracbarVals)

    if not opts.is_video:
        img_ori = cv2.imread(path)
        imgStacked, imgFinal = pipeline(img_ori)
        cv2.imshow("PipeLine",imgStacked) 
        cv2.imshow("Result", imgFinal) 
        cv2.waitKey()
    else:
        cap = cv2.VideoCapture(path)
        while True:
            success, img_ori = cap.read()
            imgStacked, imgFinal = pipeline(img_ori)
            cv2.imshow("PipeLine", imgStacked) 
            cv2.imshow("Result", imgFinal) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='./data/image/straight.jpg', help='Path to load the image or video')
    parser.add_argument("--is_video", action='store_true', help="Set true to load video")
    parser.add_argument('--frameWidth', type=int, default=480, help='The width to resize the image')
    parser.add_argument('--frameHeight', type=int, default=320, help='The height to resize the image')
    parser.add_argument('--intialTracbarVals', nargs='+', default=[36,63,13,87] ,type=int)
    parser.add_argument('--dehaze', action='store_true')
    parser.add_argument('--deshadow', action='store_true')
    parser.add_argument('--illumination', action='store_true')
    parser.add_argument('--disable_erode', action='store_true')
    opts = parser.parse_args()
    main(opts)
