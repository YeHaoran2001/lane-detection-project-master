import cv2
import math
import numpy as np
 
def DarkChannel(im,sz): # get dark channel image
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz)) # recrangle
    dark = cv2.erode(dc,kernel)
    return dark
 
def AtmLight(im,dark): # estimating the atmospheric light
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1)) # number of top 0.1% pixels
    darkvec = dark.reshape(imsz,1)
    imvec = im.reshape(imsz,3)
 
    indices = darkvec.argsort()
    indices = indices[imsz-numpx::] # first pick top 0.1% brightest pixels in the dark channel
 
    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]] # with haze
 
    A = atmsum / numpx
    return A
 
def TransmissionEstimate(im,A,sz): # (transmission: 透射率) size: size of local patch
    omega = 0.95
    im3 = np.empty(im.shape,im.dtype)
 
    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]
 
    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission
 
def Guidedfilter(im,p,r,eps): 
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r)) # 方框滤波
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p
 
    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I
 
    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I
 
    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))
 
    q = mean_a*im + mean_b
    return q
 
def TransmissionRefine(im,et):  # soft matting
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)
 
    return t
 
def Recover(im,t,A,tx = 0.1):  # recover the scene radiance to preserve a small amount of haze
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)
 
    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
 
    return res

def dehaze(img):
    I = img.astype('float64')/255
    dark = DarkChannel(I,15)
    A = AtmLight(I,dark)
    te = TransmissionEstimate(I,A,15)
    t = TransmissionRefine(img,te)
    J = Recover(I,t,A,0.1)
    J = (np.round(J*255)).astype('uint8')
    return J

if __name__ == '__main__':
    fn = './data/image/haze.jpg'
    img = cv2.imread(fn)
    print(img.shape, img.dtype)
    ret = dehaze(img)
    print(ret.shape, ret.dtype)
    print(ret)
    cv2.imshow('', ret)
    cv2.waitKey()

    