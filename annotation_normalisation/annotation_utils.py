import os
import sys
from PIL import Image
sys.path.append('..', )

from utils.gcloud_access import extract_files

def polygon_to_bounding_box(lst_vertex):

	minx, miny = float("inf"), float("inf")
	maxx, maxy = float("-inf"), float("-inf")

	if len(lst_vertex) == 0:
		raise ValueError("Can't compute bounding box of empty list")
	for vertex in lst_vertex:
		# Set min coords
		if vertex[0] < minx:
			minx = vertex[0]
		if vertex[1] < miny:
			miny = vertex[1]
		# Set max coords
		if vertex[0] > maxx:
			maxx = vertex[0]
		if vertex[1] > maxy:
			maxy = vertex[1]

	return [minx, maxx, miny, maxy]

def get_image_dimensions(imgname):
	im = Image.open(imgname)
	width, height = im.size
	return width, height

def get_yolo_coordinates(x_min, x_max, y_min, y_max, img_width, img_height):
	x1 = float(x_min)
	y1 = float(y_min)
	x2 = float(x_max)
	x2 += (1 if x2 == x1 else 0)
	y2 = float(y_max)
	y2 += (1 if y2 == y1 else 0)

	relative_x_center = ((x1 + x2) / 2) / float(img_width)
	relative_y_center = ((y1 + y2) / 2) / float(img_height)
	relative_object_width = (x2 - x1) / float(img_width)
	relative_object_height = (y2 - y1) / float(img_height)

	return relative_x_center, relative_y_center, relative_object_width, relative_object_height