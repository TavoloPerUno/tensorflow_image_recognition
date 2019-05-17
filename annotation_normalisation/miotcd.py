import os
import sys
import pandas as pd

sys.path.append('..', )

from annotation_normalisation.annotations import Annotations
from annotation_normalisation.annotation_utils import *
class Miotcd(Annotations):

	def __init__(self):
		super().__init__('miotcd')

	def normalise_annotation_row(self, dfrow):
		img_width, img_height = get_image_dimensions(
			os.path.join(self.images_folder, 'MIO-TCD-Localization', 'train',
						 '{}.jpg'.format(dfrow['image'])))
		dfrow['dataset'] = 'miotcd'
		dfrow['subset'] = 'MIO-TCD-Localization'
		dfrow['test_train_val'] = 'train'
		dfrow['folder'] = 'train'
		dfrow['filename'] = '{}.jpg'.format(dfrow['image'])
		dfrow['path'] = os.path.join('miotcd', 'MIO-TCD-Localization', 'train', '{}.jpg'.format(dfrow['image']))
		dfrow['width'] = img_width
		dfrow['height'] = img_height
		dfrow['x_min'] = float(dfrow['gt_x1'])
		dfrow['x_max'] = float(dfrow['gt_x2'])
		dfrow['y_min'] = float(dfrow['gt_y1'])
		dfrow['y_max'] = float(dfrow['gt_y2'])
		dfrow['yolo_x'], dfrow['yolo_y'], dfrow['yolo_w'], dfrow['yolo_h'] = get_yolo_coordinates(dfrow['gt_x1'],
																								  dfrow['gt_x2'],
																								  dfrow['gt_y1'],
																								  dfrow['gt_y2'],
																								  img_width, img_height)
		return dfrow[
			['dataset', 'subset', 'test_train_val', 'folder', 'filename', 'path', 'label', 'width', 'height', 'x_min',
			 'x_max', 'y_min', 'y_max', 'yolo_x', 'yolo_y', 'yolo_w', 'yolo_h']]

	def normalise_annotations(self):
		annotation_raw_file = os.path.join(self.annotations_folder, 'MIO-TCD-Localization',
										   'gt_train.csv')
		df_annotations_raw = pd.read_csv(annotation_raw_file, header=None, dtype={0: str})
		df_annotations_raw.columns = ['image', 'label', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2']

		self.df_annotations =self.df_annotations.append(df_annotations_raw.apply(self.normalise_annotation_row, axis=1),
											   ignore_index=True)

		self.df_annotations.to_csv(self.annotations_dest_name, header=True, index=False)

