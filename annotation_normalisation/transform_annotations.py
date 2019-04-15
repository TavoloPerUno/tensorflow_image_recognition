import os
import sys
import pandas as pd
import random
import cv2
import logging
import datetime
import argparse
import json
import shutil
from PIL import Image
from logging.handlers import TimedRotatingFileHandler

sys.path.append('..', )

from utils.gcloud_access import extract_files





logname = os.path.join('..', 'logs', 'tensorflow_image_recognition_annotation_normalisation {:%Y-%m-%d %H:%M:%S}.log'.format(datetime.datetime.now()))
handler = logging.FileHandler(logname, mode='a')

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logging = logging.getLogger()
logging.addHandler(handler)

dct_global_constants = dict()
dct_global_constants['compressed_annotations_folder'] =os.path.join('..', 'data', 'training', 'annotations_compressed')
dct_global_constants['annotations_folder'] = os.path.join('..', 'data', 'training', 'annotations')
dct_global_constants['meta_folder'] = os.path.join('..', 'data', 'training', 'metadata')

dct_global_constants['compressed_images_folder'] = os.path.join('..', 'data', 'training', 'images_compressed')
dct_global_constants['images_folder'] = os.path.join('..', 'data', 'training', 'images')

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

def download_data(source, is_compressed):
	if is_compressed:

		if source == "cityscapes":


			extract_files(os.path.join(dct_global_constants['compressed_annotations_folder'], source, 'gtCoarse.zip'), os.path.join(dct_global_constants['annotations_folder'], source))
			extract_files(os.path.join(dct_global_constants['compressed_annotations_folder'], source, 'gtFine_trainvaltest.zip'),
						  os.path.join(dct_global_constants['annotations_folder'], source))

			# download_from_bucket(bucket_id, os.path.join(dct_global_constants['compressed_annotations_folder'], source, 'gtCoarse.zip'), project_id, dct_global_constants['data_download_dest'], dct_global_constants['mounted_folder'], is_mounted)
			# download_from_bucket(bucket_id, os.path.join(dct_global_constants['compressed_annotations_folder'], source, 'gtFine_trainvaltest.zip'), project_id, dct_global_constants['data_download_dest'], dct_global_constants['mounted_folder'], is_mounted)

		if source == "mapillaryvistas":
			extract_files(os.path.join(dct_global_constants['compressed_annotations_folder'], source, 'mapillary-vistas-dataset_public_v1.1.zip'),
						  os.path.join(dct_global_constants['annotations_folder'], source))

			#download_from_bucket(bucket_id, os.path.join(source, 'mapillary-vistas-dataset_public_v1.1.zip'), project_id, dct_global_constants['data_download_dest'], dct_global_constants['mounted_folder'], is_mounted)


		shutil.rmtree(os.path.join(dct_global_constants['compressed_annotations_folder'], source))

	return

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

def get_normalised_miot_annotations(dfrow):
	img_width, img_height = get_image_dimensions(os.path.join(dct_global_constants['images_folder'], 'miotcd', 'MIO-TCD-Localization', 'train', '{}.jpg'.format(dfrow['image'])))
	dfrow['dataset'] = 'miotcd'
	dfrow['subset'] = 'MIO-TCD-Localization'
	dfrow['test_train_val'] = 'train'
	dfrow['folder'] = 'train'
	dfrow['filename'] ='{}.jpg'.format(dfrow['image'])
	dfrow['path'] = os.path.join('miotcd', 'MIO-TCD-Localization', 'train', '{}.jpg'.format(dfrow['image']))
	dfrow['width'] = img_width
	dfrow['height'] = img_height
	dfrow['x_min'] = float(dfrow['gt_x1'])
	dfrow['x_max'] = float(dfrow['gt_x2'])
	dfrow['y_min'] = float(dfrow['gt_y1'])
	dfrow['y_max'] = float(dfrow['gt_y2'])
	dfrow['yolo_x'], dfrow['yolo_y'], dfrow['yolo_w'], dfrow['yolo_h'] = get_yolo_coordinates(dfrow['gt_x1'],dfrow['gt_x2'], dfrow['gt_y1'],
						 dfrow['gt_y2'], img_width, img_height)
	return dfrow[['dataset', 'subset', 'test_train_val', 'folder', 'filename', 'path', 'label', 'width', 'height', 'x_min', 'x_max','y_min', 'y_max', 'yolo_x', 'yolo_y', 'yolo_w', 'yolo_h']]


def get_normalised_cityscape_annotations(dfrow):
	dfrow['dataset'] = 'cityscapes'
	dfrow['filename'] = '_'.join(dfrow['filename'].split('_', 3)[:3]) + '_leftImg8bit.png'
	dfrow['width'] = dfrow['imgWidth']
	dfrow['height'] = dfrow['imgHeight']
	lst_objects = []
	for dct_object in dfrow['objects']:
		lst_bounds = polygon_to_bounding_box(dct_object['polygon'])
		yolo_x, yolo_y, yolo_w, yolo_h = get_yolo_coordinates(lst_bounds[0], lst_bounds[1], lst_bounds[2],lst_bounds[3], dfrow['imgWidth'],dfrow['imgHeight'])
		lst_objects.append((dct_object['label'], lst_bounds[0], lst_bounds[1], lst_bounds[2],lst_bounds[3], yolo_x, yolo_y, yolo_w, yolo_h))
	dfrow['objects'] = lst_objects
	return dfrow[['dataset', 'filename', 'width', 'height', 'objects']]

def normalise_annotations(source):

	annotations_dest_name = os.path.join(dct_global_constants['meta_folder'], 'gsv_annotations.csv')

	df_annotations = pd.DataFrame(columns=['dataset', 'subset', 'test_train_val', 'folder', 'filename', 'path', 'label', 'width', 'height', 'x_min', 'x_max', 'y_min', 'y_max', 'yolo_x', 'yolo_y', 'yolo_w', 'yolo_h'])

	if os.path.isfile(annotations_dest_name):
		df_annotations = pd.read_csv(annotations_dest_name)

	if source == "cityscapes":
		for cat in ['train', 'test', 'val', 'train_extra']:
			base_folder = os.path.join(dct_global_constants['annotations_folder'], 'cityscapes', (os.path.join('gtFine_trainvaltest', 'gtFine', cat) if (cat != 'train_extra') else os.path.join('gtCoarse', 'gtCoarse', cat)))
			for folder_path in os.scandir(base_folder):
				folder = os.path.basename(folder_path)
				if not folder_path.name.startswith('.') and folder_path.is_dir():
					lst_json_files = [pos_json for pos_json in os.listdir(os.path.join(base_folder, folder)) if pos_json.endswith('.json')]
					dct_annotations = {}
					for json_file in lst_json_files:
						with open(os.path.join(base_folder, folder, json_file), "r") as inputjson:
							dct_annotations[json_file] = json.load(inputjson)

					df_annotations_raw = pd.DataFrame(dct_annotations)
					df_annotations_raw  = df_annotations_raw .T
					df_annotations_raw ['filename'] = df_annotations_raw .index
					df_annotations_raw .reset_index(level=0, inplace=True)
					df_annotations_modified = df_annotations_raw.apply(get_normalised_cityscape_annotations, axis=1)
					
					df_annotations_modified = df_annotations_modified.objects.apply(pd.Series).merge(df_annotations_modified, right_index = True, left_index = True).drop(["objects"], axis = 1).melt(id_vars = ['dataset', 'filename', 'width', 'height'], value_name = "object").drop("variable", axis = 1).dropna()
					lst_new_columns = ['label', 'x_min', 'x_max','y_min', 'y_max', 'yolo_x', 'yolo_y', 'yolo_w', 'yolo_h']
					for n,col in enumerate(lst_new_columns ):
						df_annotations_modified[col] = df_annotations_modified['object'].apply(lambda anno: anno[n])
					df_annotations_modified = df_annotations_modified.drop('object',axis=1)
					df_annotations_modified['subset'] =  ('gtFine' if (cat != 'train_extra') else 'gtCoarse')
					df_annotations_modified['test_train_val'] =  cat
					df_annotations_modified['folder'] = folder
					df_annotations_modified['path'] = df_annotations_modified['filename'].apply(lambda x: os.path.join('cityscapes', 'leftImg8bit', cat, folder, x))
                                                                                                                                                                         
					df_annotations = df_annotations.append(df_annotations_modified[['dataset', 'subset', 'test_train_val', 'folder', 'filename', 'path', 'label', 'width', 'height', 'x_min', 'x_max','y_min', 'y_max', 'yolo_x', 'yolo_y', 'yolo_w', 'yolo_h']],
                                       ignore_index=True)
					
					logging.info("Saved annotations from {0}".format(os.path.join(base_folder, folder)))
					df_annotations.to_csv(annotations_dest_name, header=True, index=False)

	if source == "miotcd":
		annotation_raw_file = os.path.join(dct_global_constants['annotations_folder'], 'miotcd', 'MIO-TCD-Localization',  'gt_train.csv')
		df_annotations_raw = pd.read_csv(annotation_raw_file, header=None, dtype={0: str})
		df_annotations_raw.columns = ['image', 'label', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2']

		df_annotations = df_annotations.append(df_annotations_raw.apply(get_normalised_miot_annotations, axis=1),
                                                       ignore_index=True)

		df_annotations.to_csv(annotations_dest_name, header=True, index=False)


def main(argv):
	parser = argparse.ArgumentParser(description='Convert annotation format')

	parser.add_argument('s', type=str,
						help='Dataset name. Supported types: cityscapes, mapillaryvistas')

	parser.add_argument('c', type=int,
						help='Are input files compressed?')

	parser.add_argument('-t', type=str,
						help='annotation_type')

	parser.add_argument('-o', type=str,
						help='output_folder')

	parser.add_argument('-n', type=str,
						help='ignored classes')


	args = parser.parse_args()

	source = args.s

	is_compressed = (args.c == 1)

	#download_data(source, is_compressed)
	normalise_annotations(source)


if __name__ == '__main__':
	main(sys.argv[1:])
