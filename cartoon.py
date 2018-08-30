import cv2
import numpy 
from PIL import ImageFilter,Image,ImageDraw
Filters=[ImageFilter.BLUR,ImageFilter.CONTOUR,ImageFilter.EDGE_ENHANCE,ImageFilter.EDGE_ENHANCE_MORE,ImageFilter.EMBOSS,ImageFilter.FIND_EDGES,ImageFilter.SMOOTH,ImageFilter.SMOOTH_MORE,ImageFilter.SHARPEN,ImageFilter.DETAIL]

def cartoon(img,num):
	img=Image.fromarray(img)
	#img = Image.open("hello.png")
	img=img.filter(Filters[num])
	img = numpy.asarray(img)
	return img
