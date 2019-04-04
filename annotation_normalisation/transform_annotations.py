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
from logging.handlers import TimedRotatingFileHandler

sys.path.append('..', )

from utils.gcloud_access import extract_files





logname = os.path.join('..', 'logs', 'tensorflow_image_recognition {:%Y-%m-%d %H:%M:%S}.log'.format(datetime.datetime.now()))
handler = logging.FileHandler(logname, mode='a')

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logging = logging.getLogger()
logging.addHandler(handler)

dct_global_constants = dict()
dct_global_constants['compressed_annotations_folder'] =os.path.join('..', 'mounted_bucket', 'annotations_compressed')
dct_global_constants['annotations_folder'] = os.path.join('..', 'mounted_bucket', 'annotations')
dct_global_constants['meta_folder'] = os.path.join('..', 'mounted_bucket', 'metadata')

dct_global_constants['compressed_images_folder'] = os.path.join('..', 'mounted_bucket', 'images_compressed')
dct_global_constants['images_folder'] = os.path.join('..', 'mounted_bucket', 'images')
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

def get_bounding_boxes(source):

	annotations_dest_name = os.path.join(dct_global_constants['meta_folder'], 'annotations.csv')

	df_annotations = pd.DataFrame(columns=['dataset', 'subset', 'test_train_val', 'folder', 'filename', 'path', 'label', 'width', 'height', 'x_min', 'x_max', 'y_min', 'y_max', 'yolo_x', 'yolo_y', 'yolo_h', 'yolo_w'])

	if os.path.isfile(annotations_dest_name):
		df_annotations = pd.read_csv(annotations_dest_name)

	if source == "cityscapes":
		for cat in ['train', 'test', 'val', 'train_extra']:
			base_folder = os.path.join(dct_global_constants['data_download_dest'], (os.path.join('gtFine_trainvaltest', 'gtFine', cat) if (cat != 'train_extra') else os.path.join('gtCoarse', 'gtCoarse', cat)))
			for folder_path in os.scandir(base_folder):
				folder = os.path.basename(folder_path)
				if not folder_path.name.startswith('.') and folder_path.is_dir():
					for file_path in os.scandir(os.path.join(base_folder, folder)):
						file = os.path.basename(file_path)
						if not file_path.name.startswith('.') and file_path.is_file() and os.path.splitext(file)[1].lower() == '.json':
							with open(os.path.join(base_folder, folder, file)) as f:
								dct_annotation = json.load(f)

							for dct_object in dct_annotation['objects']:
								lst_bounds = polygon_to_bounding_box(dct_object['polygon'])

								df_annotations = df_annotations.append(pd.DataFrame({'dataset': 'cityscapes',
																					 'subset': ('gtFine' if (cat != 'train_extra') else 'gtCoarse'),
																					 'test_train_val': cat,
																					 'folder': folder,
																					 'filename': file,
																					 'path': os.path.join('cityscapes', ('gtFine_trainvaltest' if (cat != 'train_extra') else 'gtCoarse'), ('gtFine' if (cat != 'train_extra') else 'gtCoarse'), cat, folder, file),
																					 'label': dct_object['label'],
																					 'width': dct_annotation['imgHeight'],
																					 'height': dct_annotation['imgWidth'],
																					 'x_min': lst_bounds[0],
																					 'x_max': lst_bounds[1],
																					 'y_min': lst_bounds[2],
																					 'y_max': lst_bounds[3],
																					 'yolo_x': (lst_bounds[0] + lst_bounds[1]) / (2 * float(dct_annotation['imgHeight'])),
																					 'yolo_y': (lst_bounds[2] + lst_bounds[3]) / (2 * float(dct_annotation['imgWidth'])),
																					 'yolo_w': (lst_bounds[1] - lst_bounds[0])/ float(dct_annotation['imgHeight']),
																					 'yolo_h': (lst_bounds[3] - lst_bounds[2])/ float(dct_annotation['imgWidth']),

																					 }, index=[0]),
																	   ignore_index=True)

								logging.info("Saved annotations from {0}".format(file_path))


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

	download_data(source, is_compressed)
	get_bounding_boxes(source)


if __name__ == '__main__':
	main(sys.argv[1:])