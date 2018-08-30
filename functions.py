import cv2
import dlib
import numpy as np
import pickle
import sys
import math
from cartoon import *
from functions import *



SCALE_RATE_SET = 0.2 # Set scale rate for eyes
V3_DIC = {'00':22}  # In mode v3, the mode name with its correspond number of frams.

'''
This part is for face detection
'''
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11


FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
# OVERLAY_POINTS = [
#     LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
#     NOSE_POINTS + MOUTH_POINTS,
# ]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
def get_landmarks(im):
    rects = detector(im, 1)
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

# generate landmarks for white
def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in range(len(landmarks)):
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return im
    
def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))


''' 
These two function are from reference about local trasition and local scale of a image.
They base on the formula from reference. when inputs an destinate image pixel point and related 
parameters, it will return the correspond pixel point in source image.
'''
def pointTransition(xPoint, mousePoint, center, rMax):
    xcDistance = (xPoint[0] - center[0])**2 + (xPoint[1] - center[1])**2
    mcDistance = (mousePoint[0] - center[0])**2 + (mousePoint[1] - center[1])**2
    i = rMax ** 2 - xcDistance
    j = (i/(i+mcDistance))**2
    tmp = [j*(mousePoint[0] - center[0]), j*(mousePoint[1] - center[1])]
    return (int(xPoint[0]-tmp[0]),int(xPoint[1]-tmp[1]))

def pointScale(xPoint,mRate,center,rMax):
    tmp1 = (xPoint[0] - center[0])**2 +(xPoint[1] - center[1])**2
    tmp1 = math.sqrt(tmp1)
    tmp1 = 1-(tmp1/rMax - 1)**2 * mRate
    return (int(center[0]+tmp1*(xPoint[0] - center[0])),int(center[1]+tmp1*(xPoint[1] - center[1])))

''' This function is to locate the eyes area and call local scale function(pointScale)'''
def biggerEye(img, landmarks):
    copy = np.zeros(img.shape, dtype = img.dtype)
    copy[:,:,:] = img[:,:,:]
    # left eye
    center = (int((landmarks[36,1]+landmarks[39,1])/2),int((landmarks[36,0]+landmarks[39,0])/2))
    rMax = abs(int((landmarks[36,0]-landmarks[39,0])))*2
    mRate = SCALE_RATE_SET
    for i in range(center[0]-rMax,center[0]+rMax):
        for j in range(center[1]-rMax,center[1]+rMax):
            a = i - center[0]
            b = j - center[1]
            if a**2 + b**2 < rMax**2:

                x,y = pointScale((i,j),mRate,center,rMax)
                if x<img.shape[0] and y<img.shape[1] and x >=0 and y >=0:
                    if i<img.shape[0] and j<img.shape[1] and i >=0 and j >=0:
                        copy[i,j,:] = img[x,y,:]

    # Right eye
    center = (int((landmarks[42,1]+landmarks[45,1])/2),int((landmarks[42,0]+landmarks[45,0])/2))
    rMax = abs(int((landmarks[42,0]-landmarks[45,0])))*2
    for i in range(center[0]-rMax,center[0]+rMax):
        for j in range(center[1]-rMax,center[1]+rMax):
            a = i - center[0]
            b = j - center[1]
            if a**2 + b**2 < rMax**2:
                x,y = pointScale((i,j),mRate,center,rMax)
                if x<img.shape[0] and y<img.shape[1] and x >=0 and y >=0:
                    if i<img.shape[0] and j<img.shape[1] and i >=0 and j >=0:
                        copy[i,j,:] = img[x,y,:]
    return copy


# Some function for path
def modes(name):
    landmarks_path = 'sticker_train_set/Effect '+name+'.txt'
    im3_path = 'COMP9517 Pics/Sticker '+name+'.png'
    im2_path = 'COMP9517 Pics/Effect '+name+'.png'
    return im3_path,landmarks_path,im2_path

def get_v3_paths(argument):
    paths = ['Animate/Animation '+argument+' '+str(i)+'.png' for i in range(V3_DIC[argument])]
    return paths



def demo(mode_dic):
    if mode_dic['v1'] != None:
        im3_path, landmarks_path, im2_path = modes(mode_dic['v1'])
        im2,landmarks2 = read_im_and_landmarks(im2_path)
        im3 = cv2.imread(im3_path,-1)
    if mode_dic['v3'] != None:
        v3_paths = get_v3_paths(mode_dic['v3'])
        v3_couter = 0
        v3_imgs = [cv2.pyrDown(cv2.imread(v3_paths[i],-1)) for i in range(V3_DIC[mode_dic['v3']])]

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    im1 = cv2.pyrDown(frame)
    a,b,c = im1.shape
    dshape = (im1.shape[0],im1.shape[1],4)
    dst = np.zeros(dshape, dtype=im1.dtype)
    dst.fill(255)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        im1 = cv2.pyrDown(frame)
        '''following steps are for transforming im1 which is RGB image to image with alpha channel'''        
        if mode_dic['v2']!= None:
            # pass
            im1 = cartoon(im1,int(mode_dic['v2']))
        if mode_dic['v1'] != None and mode_dic['v3'] != None:
            rects = detector(im1, 1)
            if len(rects) != 0:
                List_of_landmarks = []
                for index, face in enumerate(rects):
                    landmarks1 = np.matrix([[p.x, p.y] for p in predictor(im1, face).parts()])
                    List_of_landmarks.append(landmarks1)
                    '''calculate the transformation matrix M'''
                    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                                   landmarks2[ALIGN_POINTS])

                    '''affine function'''
                    warped_im2 = warp_im(im3, M, dshape)
                    '''combine two images'''
                    tmp2 = (warped_im2[:,:,3] == 255)
                    dst[:,:,:3]= im1[:,:,:]
                    dst[tmp2] = warped_im2[tmp2]
                    if v3_couter >= V3_DIC[mode_dic['v3']]:
                        v3_couter = 0
                    v3_im = v3_imgs[v3_couter]
                    tmp2 = (v3_im[:,:,3]==255)
                    dst[tmp2] = v3_im[tmp2]
                    im1[:,:,:] = dst[:,:,:3] 
                    v3_couter += 1     
                    im1 = biggerEye(im1,landmarks1)
        elif mode_dic['v1'] != None:
            '''this function belongs to face detection, in order to get landmarks1'''
            rects = detector(im1, 1)
            if len(rects) != 0:
                List_of_landmarks = []
                for index, face in enumerate(rects):
                    landmarks1 = np.matrix([[p.x, p.y] for p in predictor(im1, face).parts()])
                    List_of_landmarks.append(landmarks1)
                    '''calculate the transformation matrix M'''
                    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                                   landmarks2[ALIGN_POINTS])

                    '''affine function'''
                    warped_im2 = warp_im(im3, M, dshape)
                    '''combine two images'''
                    tmp2 = (warped_im2[:,:,3] == 255)
                    dst[:,:,:3]= im1[:,:,:]
                    dst[tmp2] = warped_im2[tmp2]
                    im1[:,:,:] = dst[:,:,:3]     
                    im1 = biggerEye(im1,landmarks1)
        # raise NoFaces
        # Display the resulting frame
        elif mode_dic['v3']!= None:
            if v3_couter >= V3_DIC[mode_dic['v3']]:
                v3_couter = 0
            dst[:,:,:3]= im1[:,:,:]
            # v3_im = cv2.imread(v3_paths[v3_couter],-1)
            # v3_im = cv2.pyrDown(v3_im)
            v3_im = v3_imgs[v3_couter]
            tmp2 = (v3_im[:,:,3]==255)
            dst[tmp2] = v3_im[tmp2]
            im1[:,:,:] = dst[:,:,:3] 
            v3_couter += 1

        im1 = cv2.pyrUp(im1)
        # im1 = cv2.pyrUp(im1)
        cv2.imshow('frame',im1)
        if cv2.waitKey(1) & 0xFF == ord('b'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



def run_camera(mode_dic):
    if mode_dic['v1'] != None: # if v1 activated
        '''
        Find the correspond Sticker and Effect path
                mode_dic['v1'] range from '00' to '17' 
        im1 read from camera
        im2 is Effect image
        im3 is Sticker with alpha channel
        landmask1 is landmask from im1
        landmask2 is landmask from im2
        '''
        im3_path, landmarks_path, im2_path = modes(mode_dic['v1'])
        im2,landmarks2 = read_im_and_landmarks(im2_path)
        im3 = cv2.imread(im3_path,-1)

    if mode_dic['v3'] != None:
        '''v3 mode should load a image sequence first before using camera'''
        v3_paths = get_v3_paths(mode_dic['v3'])
        v3_couter = 0 # This classify which image in sequence should be used 
        v3_imgs = [cv2.pyrDown(cv2.imread(v3_paths[i],-1)) for i in range(V3_DIC[mode_dic['v3']])]

    # run camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    im1 = cv2.pyrDown(frame)
    # dst is a buffer space to store intermediat image. It has alpha channel
    a,b,c = im1.shape
    dshape = (im1.shape[0],im1.shape[1],4)
    dst = np.zeros(dshape, dtype=im1.dtype)
    dst.fill(255)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # reduce the size of image in order to process quicker
        im1 = cv2.pyrDown(frame)
        '''following steps are for transforming im1 which is RGB image to image with alpha channel'''        
        if mode_dic['v2']!= None and mode_dic['v1']== None and mode_dic['v3']==None:
            # apply filter
            im1 = cartoon(im1,int(mode_dic['v2']))
        if mode_dic['v1'] != None and mode_dic['v3'] != None:
            rects = detector(im1, 1)
            if len(rects) != 0:
                List_of_landmarks = []
                for index, face in enumerate(rects):
                    landmarks1 = np.matrix([[p.x, p.y] for p in predictor(im1, face).parts()])
                    List_of_landmarks.append(landmarks1)
                    '''calculate the transformation matrix M'''
                    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                                   landmarks2[ALIGN_POINTS])

                    '''affine function'''
                    warped_im2 = warp_im(im3, M, dshape)
                    '''combine two images'''
                    tmp2 = (warped_im2[:,:,3] == 255)
                    # if mode_dic['v2']!= None:
                    #     # apply filter
                    #     t = cartoon(im1,int(mode_dic['v2']))
                    #     im1 = np.zeros(t.shape, dtype=im1.dtype)
                    dst[:,:,:3]= im1[:,:,:]
                    dst[tmp2] = warped_im2[tmp2]

                    if v3_couter >= V3_DIC[mode_dic['v3']]:
                        v3_couter = 0
                    v3_im = v3_imgs[v3_couter]
                    tmp2 = (v3_im[:,:,3]==255)
                    dst[tmp2] = v3_im[tmp2]
                    im1[:,:,:] = dst[:,:,:3] 
                    v3_couter += 1     
                    im1 = biggerEye(im1,landmarks1)
        elif mode_dic['v1'] != None:
            '''this function belongs to face detection, in order to get landmarks1'''
            rects = detector(im1, 1)
            if len(rects) != 0:
                List_of_landmarks = []
                for index, face in enumerate(rects):
                    landmarks1 = np.matrix([[p.x, p.y] for p in predictor(im1, face).parts()])
                    List_of_landmarks.append(landmarks1)
                    '''calculate the transformation matrix M'''
                    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                                   landmarks2[ALIGN_POINTS])

                    '''affine function'''
                    warped_im2 = warp_im(im3, M, dshape)
                    '''combine two images'''
                    tmp2 = (warped_im2[:,:,3] == 255)
                    # if mode_dic['v2']!= None:
                    #     # apply filter

                    #     t = cartoon(im1,int(mode_dic['v2']))
                    #     im1 = np.zeros(t.shape, dtype=im1.dtype)
                    dst[:,:,:3]= im1[:,:,:]
                    dst[tmp2] = warped_im2[tmp2]
                    im1[:,:,:] = dst[:,:,:3]     
                    im1 = biggerEye(im1,landmarks1)
        # raise NoFaces
        # Display the resulting frame
        elif mode_dic['v3']!= None:
            if v3_couter >= V3_DIC[mode_dic['v3']]:
                v3_couter = 0
            dst[:,:,:3]= im1[:,:,:]
            # v3_im = cv2.imread(v3_paths[v3_couter],-1)
            # v3_im = cv2.pyrDown(v3_im)
            v3_im = v3_imgs[v3_couter]
            tmp2 = (v3_im[:,:,3]==255)
            dst[tmp2] = v3_im[tmp2]
            im1[:,:,:] = dst[:,:,:3] 
            v3_couter += 1
        if mode_dic['smallerface']:
            im1=smallerFace(im1)
        im1 = cv2.pyrUp(im1)
        # im1 = cv2.pyrUp(im1)
        cv2.imshow('frame',im1)
        if cv2.waitKey(1) & 0xFF == ord('b'):
            if mode_dic['v2'] != None:
                frame = cartoon(frame,int(mode_dic['v2']))
            a,b,c = frame.shape
            dshape = (frame.shape[0],frame.shape[1],4)
            dst = np.zeros(dshape, dtype=frame.dtype)
            dst.fill(255)
            if mode_dic['v1'] != None:
                for landmarks1 in List_of_landmarks:
                    landmarks1 = np.multiply(landmarks1,2)
                    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                                   landmarks2[ALIGN_POINTS])                    
                    dst[:,:,:3]= frame[:,:,:]
                    '''affine function'''
                    warped_im2 = warp_im(im3, M, dshape)
                    '''combine two images'''
                    tmp2 = (warped_im2[:,:,3] == 255)
                    dst[tmp2] = warped_im2[tmp2]
                    frame[:,:,:] = dst[:,:,:3]
                    frame = biggerEye(frame,landmarks1)
            if mode_dic['v3'] != None:
                v3_im = cv2.imread(v3_paths[v3_couter],-1)
                # print(v3_paths[v3_couter])
                dst[:,:,:3] = frame[:,:,:]
                tmp2 = (v3_im[:,:,3]==255)
                dst[tmp2] = v3_im[tmp2]
                frame[:,:,:] = dst[:,:,:3] 
            if mode_dic['smallerface']:
                frame=smallerFace(frame)
            cv2.imwrite('output.jpg', frame)
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def dynamicSticker(argument):
    im2_path = 'Archive/Effect '+argument+'.png'
    im3_paths = ['Archive/Sticker '+argument+' '+str(i)+'.png' for i in range(4)]
    im2,landmarks2 = read_im_and_landmarks(im2_path)
    im3s = [cv2.imread(i,-1) for i in im3_paths]
    im3_couter = 0 # This classify which image in sequence should be used 
    # run camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    im1 = cv2.pyrDown(frame)
    # dst is a buffer space to store intermediat image. It has alpha channel
    a,b,c = im1.shape
    dshape = (im1.shape[0],im1.shape[1],4)
    dst = np.zeros(dshape, dtype=im1.dtype)
    dst.fill(255)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # reduce the size of image in order to process quicker
        im1 = cv2.pyrDown(frame)
        '''following steps are for transforming im1 which is RGB image to image with alpha channel'''        
 
        '''this function belongs to face detection, in order to get landmarks1'''
        rects = detector(im1, 1)
        if len(rects) != 0:
            List_of_landmarks = []
            for index, face in enumerate(rects):
                landmarks1 = np.matrix([[p.x, p.y] for p in predictor(im1, face).parts()])
                List_of_landmarks.append(landmarks1)
                '''calculate the transformation matrix M'''
                M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                               landmarks2[ALIGN_POINTS])

                '''affine function'''
                warped_im2 = warp_im(im3s[im3_couter], M, dshape)
                '''combine two images'''
                tmp2 = (warped_im2[:,:,3] == 255)
                dst[:,:,:3]= im1[:,:,:]
                dst[tmp2] = warped_im2[tmp2]
                im1[:,:,:] = dst[:,:,:3]     
                im1 = biggerEye(im1,landmarks1)
            im3_couter += 1
            if im3_couter >= 4:
                im3_couter = 0

        # raise NoFaces
        # Display the resulting frame
        im1 = cv2.pyrUp(im1)
        # im1 = cv2.pyrUp(im1)
        cv2.imshow('frame',im1)
        if cv2.waitKey(1) & 0xFF == ord('b'):
            a,b,c = frame.shape
            dshape = (frame.shape[0],frame.shape[1],4)
            dst = np.zeros(dshape, dtype=frame.dtype)
            dst.fill(255)

            for landmarks1 in List_of_landmarks:
                landmarks1 = np.multiply(landmarks1,2)
                M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                               landmarks2[ALIGN_POINTS])                    
                dst[:,:,:3]= frame[:,:,:]
                '''affine function'''
                warped_im2 = warp_im(im3s[im3_couter], M, dshape)
                '''combine two images'''
                tmp2 = (warped_im2[:,:,3] == 255)
                dst[:,:,:3]= im1[:,:,:]
                dst[tmp2] = warped_im2[tmp2]
                im1[:,:,:] = dst[:,:,:3]     
                im1 = biggerEye(im1,landmarks1)
                im3_couter += 1
                if im3_couter >= 4:
                    im3_couter = 0

            cv2.imwrite('output.jpg', frame)
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def smallerFace(img):
    RATE = 0.5
    # NOSE_TO_JAWLINE_RATE = 1
    # CENTER_RATE = 0.75
    # M_RATE = 0.1
    copy = np.zeros(img.shape, dtype = img.dtype)
    draft = np.zeros(img.shape, dtype = img.dtype)
    copy[:,:,:] = img[:,:,:]
    draft[:,:,:] = img[:,:,:]

    center = (int(img.shape[0]/2),int(img.shape[1]/2))
    mousePoint = (center[0],center[1]+int(center[0]/2))
    rMax = int(center[0]/2)

    for i in range(center[0]-rMax,center[0]+rMax):
        for j in range(center[1]-rMax,center[1]+rMax):
            a = i - center[0]
            b = j - center[1]
            if a**2 + b**2 < rMax**2:
                # c = center[0]+mRate*a
                # d = center[1]+mRate*b
                x,y = pointTransition((i,j),mousePoint,center,rMax)
                if x <img.shape[0] and x>=0 and y< img.shape[1] and y >=0:
                    copy[i,j,:] = draft[x,y,:]
    return copy

def run_image(mode_dic,argument):
    if mode_dic['v1'] != None: # if v1 activated
        '''
        Find the correspond Sticker and Effect path
                mode_dic['v1'] range from '00' to '17' 
        im1 read from camera
        im2 is Effect image
        im3 is Sticker with alpha channel
        landmask1 is landmask from im1
        landmask2 is landmask from im2
        '''
        im3_path, landmarks_path, im2_path = modes(mode_dic['v1'])
        im2,landmarks2 = read_im_and_landmarks(im2_path)
        im3 = cv2.imread(im3_path,-1)

    if mode_dic['v3'] != None:
        '''v3 mode should load a image sequence first before using camera'''
        v3_paths = get_v3_paths(mode_dic['v3'])
        v3_couter = 0 # This classify which image in sequence should be used 
        v3_imgs = [cv2.pyrDown(cv2.imread(v3_paths[i],-1)) for i in range(V3_DIC[mode_dic['v3']])]

    # run camera
    # cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()

    # im1 = cv2.pyrDown(frame)
    im1 = cv2.imread(argument)
    # dst is a buffer space to store intermediat image. It has alpha channel
    a,b,c = im1.shape
    dshape = (im1.shape[0],im1.shape[1],4)
    dst = np.zeros(dshape, dtype=im1.dtype)
    dst.fill(255)

    # while(True):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     # reduce the size of image in order to process quicker
    #     im1 = cv2.pyrDown(frame)
    '''following steps are for transforming im1 which is RGB image to image with alpha channel'''        
    if mode_dic['v2']!= None:
        # apply filter
        im1 = cartoon(im1,int(mode_dic['v2']))
    if mode_dic['v1'] != None and mode_dic['v3'] != None:
        rects = detector(im1, 1)
        if len(rects) != 0:
            List_of_landmarks = []
            for index, face in enumerate(rects):
                landmarks1 = np.matrix([[p.x, p.y] for p in predictor(im1, face).parts()])
                List_of_landmarks.append(landmarks1)
                '''calculate the transformation matrix M'''
                M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                               landmarks2[ALIGN_POINTS])

                '''affine function'''
                warped_im2 = warp_im(im3, M, dshape)
                '''combine two images'''
                tmp2 = (warped_im2[:,:,3] == 255)
                dst[:,:,:3]= im1[:,:,:]
                dst[tmp2] = warped_im2[tmp2]
                if v3_couter >= V3_DIC[mode_dic['v3']]:
                    v3_couter = 0
                v3_im = v3_imgs[v3_couter]
                tmp2 = (v3_im[:,:,3]==255)
                dst[tmp2] = v3_im[tmp2]
                im1[:,:,:] = dst[:,:,:3] 
                v3_couter += 1     
                im1 = biggerEye(im1,landmarks1)
    elif mode_dic['v1'] != None:
        '''this function belongs to face detection, in order to get landmarks1'''
        rects = detector(im1, 1)
        if len(rects) != 0:
            List_of_landmarks = []
            for index, face in enumerate(rects):
                landmarks1 = np.matrix([[p.x, p.y] for p in predictor(im1, face).parts()])
                List_of_landmarks.append(landmarks1)
                '''calculate the transformation matrix M'''
                M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                               landmarks2[ALIGN_POINTS])

                '''affine function'''
                warped_im2 = warp_im(im3, M, dshape)
                '''combine two images'''
                tmp2 = (warped_im2[:,:,3] == 255)
                dst[:,:,:3]= im1[:,:,:]
                dst[tmp2] = warped_im2[tmp2]
                im1[:,:,:] = dst[:,:,:3]     
                im1 = biggerEye(im1,landmarks1)
    # raise NoFaces
    # Display the resulting frame
    elif mode_dic['v3']!= None:
        if v3_couter >= V3_DIC[mode_dic['v3']]:
            v3_couter = 0
        dst[:,:,:3]= im1[:,:,:]
        # v3_im = cv2.imread(v3_paths[v3_couter],-1)
        # v3_im = cv2.pyrDown(v3_im)
        v3_im = v3_imgs[v3_couter]
        tmp2 = (v3_im[:,:,3]==255)
        dst[tmp2] = v3_im[tmp2]
        im1[:,:,:] = dst[:,:,:3] 
        v3_couter += 1
    if mode_dic['smallerface']:
        im1=smallerFace(im1)
    # im1 = cv2.pyrUp(im1)
    return im1
    # # When everything done, release the capture
    # cv2.destroyAllWindows()