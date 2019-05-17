import numpy as np
import os
import sys
import json
import pandas as pd
import cv2

sys.path.append('..', )

from annotation_normalisation.annotations import Annotations
from annotation_normalisation.annotation_utils import *

class MapillaryVistas(Annotations):
	def __init__(self):
		super().__init__('mapillaryvistas')

	def normalise_annotation_row(self, img_filename, path, cat, dctlst_label):

		img = Image.open(os.path.join(path, img_filename))
		np_instance_image = np.array(img, dtype=np.uint16)

		img_width, img_height = img.size

		dctlst_objects = []

		np_label_pixel_info = np.array(np_instance_image / 256, dtype=np.uint8)
		np_instance_pixel_info = np.array(np_instance_image % 256, dtype=np.uint8)

		np_present_labels = np.unique(np_label_pixel_info)
		np_present_instances = np.unique(np_instance_pixel_info)

		for label_id in np_present_labels:
			if dctlst_label[label_id]["instances"] == True:
				for instance_id in np_present_instances:
					np_concerned_pixels = np.where(np.logical_and(np_label_pixel_info == label_id, np_instance_pixel_info==instance_id))

					if np_concerned_pixels[0].shape[0] > 0:
						bbox = np.min(np_concerned_pixels[0]), np.max(np_concerned_pixels[0]), np.min(np_concerned_pixels[1]), np.max(np_concerned_pixels[1])

						x_min = np.min(np_concerned_pixels[1])
						x_max = np.max(np_concerned_pixels[1])
						y_min = np.min(np_concerned_pixels[0])
						y_max = np.max(np_concerned_pixels[0])

						dct_object = dict()

						dct_object['dataset'] = 'mapillaryvistas'
						dct_object['subset'] = ''
						dct_object['test_train_val'] = cat
						dct_object['folder'] = ''
						dct_object['filename'] = os.path.splitext(img_filename)[0] + '.jpg'
						dct_object['path'] = os.path.join('mapillaryvistas', cat, 'images', os.path.splitext(img_filename)[0] + '.jpg')
						dct_object['width'] = img_width
						dct_object['height'] = img_height

						dct_object['yolo_x'], dct_object['yolo_y'], dct_object['yolo_w'], dct_object['yolo_h'] = get_yolo_coordinates(x_min, x_max, y_min, y_max, img_width, img_height)
						dct_object.update({'label': dctlst_label[label_id]['readable'].lower(),
										   'x_min': x_min,
										   'x_max': x_max,
										   'y_min': y_min,
										   'y_max': y_max})

						dctlst_objects.append(dct_object)



		return pd.DataFrame(dctlst_objects)[['dataset', 'subset', 'test_train_val', 'folder', 'filename', 'path', 'label', 'width', 'height', 'x_min',
			 'x_max', 'y_min', 'y_max', 'yolo_x', 'yolo_y', 'yolo_w', 'yolo_h']]

	def normalise_annotations(self):
		with open(os.path.join(self.annotations_folder, 'config.json')) as config_file:
			dct_config = json.load(config_file)

		dctlst_label = dct_config['labels']
		for cat in ['training', 'validation']:
			for filename in os.listdir(os.path.join(self.annotations_folder, cat, 'instances')):
				if os.path.splitext(filename)[1].lower() == '.png':
					self.df_annotations = self.df_annotations.append(self.normalise_annotation_row(filename, os.path.join(self.annotations_folder, cat, 'instances'), cat, dctlst_label))

		self.df_annotations.to_csv(self.annotations_dest_name, header=True, index=False)
