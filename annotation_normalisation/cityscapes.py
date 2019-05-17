import os
import sys
import json
import pandas as pd
import logging

sys.path.append('..', )

from annotation_normalisation.annotations import Annotations
from annotation_normalisation.annotation_utils import *
class Cityscapes(Annotations):

	def __init__(self):
		super().__init__('cityscapes')
		self.lst_compressed_annotations_file = ['gtCoarse.zip', 'gtFine_trainvaltest.zip']

	def normalise_annotation_row(self, dfrow):
		dfrow['dataset'] = 'cityscapes'
		dfrow['filename'] = '_'.join(dfrow['filename'].split('_', 3)[:3]) + '_leftImg8bit.png'
		dfrow['width'] = dfrow['imgWidth']
		dfrow['height'] = dfrow['imgHeight']
		lst_objects = []
		for dct_object in dfrow['objects']:
			lst_bounds = polygon_to_bounding_box(dct_object['polygon'])
			yolo_x, yolo_y, yolo_w, yolo_h = get_yolo_coordinates(lst_bounds[0], lst_bounds[1], lst_bounds[2],
																  lst_bounds[3], dfrow['imgWidth'], dfrow['imgHeight'])
			lst_objects.append((dct_object['label'], lst_bounds[0], lst_bounds[1], lst_bounds[2], lst_bounds[3], yolo_x,
								yolo_y, yolo_w, yolo_h))
		dfrow['objects'] = lst_objects
		return dfrow[['dataset', 'filename', 'width', 'height', 'objects']]

	def normalise_annotations(self):
		for cat in ['train', 'test', 'val', 'train_extra']:
			base_folder = os.path.join(self.annotations_folder, (
				os.path.join('gtFine_trainvaltest', 'gtFine', cat) if (cat != 'train_extra') else os.path.join(
					'gtCoarse', 'gtCoarse', cat)))
			for folder_path in os.scandir(base_folder):
				folder = os.path.basename(folder_path)
				if not folder_path.name.startswith('.') and folder_path.is_dir():
					lst_json_files = [pos_json for pos_json in os.listdir(os.path.join(base_folder, folder)) if
									  pos_json.endswith('.json')]
					dct_annotations = {}
					for json_file in lst_json_files:
						with open(os.path.join(base_folder, folder, json_file), "r") as inputjson:
							dct_annotations[json_file] = json.load(inputjson)

					df_annotations_raw = pd.DataFrame(dct_annotations)
					df_annotations_raw = df_annotations_raw.T
					df_annotations_raw['filename'] = df_annotations_raw.index
					df_annotations_raw.reset_index(level=0, inplace=True)
					df_annotations_modified = df_annotations_raw.apply(self.normalise_annotation_row, axis=1)

					df_annotations_modified = df_annotations_modified.objects.apply(pd.Series).merge(
						df_annotations_modified, right_index=True, left_index=True).drop(["objects"], axis=1).melt(
						id_vars=['dataset', 'filename', 'width', 'height'], value_name="object").drop("variable",
																									  axis=1).dropna()
					lst_new_columns = ['label', 'x_min', 'x_max', 'y_min', 'y_max', 'yolo_x', 'yolo_y', 'yolo_w',
									   'yolo_h']
					for n, col in enumerate(lst_new_columns):
						df_annotations_modified[col] = df_annotations_modified['object'].apply(lambda anno: anno[n])
					df_annotations_modified = df_annotations_modified.drop('object', axis=1)
					df_annotations_modified['subset'] = ('gtFine' if (cat != 'train_extra') else 'gtCoarse')
					df_annotations_modified['test_train_val'] = cat
					df_annotations_modified['folder'] = folder
					df_annotations_modified['path'] = df_annotations_modified['filename'].apply(
						lambda x: os.path.join('cityscapes', 'leftImg8bit', cat, folder, x))

					self.df_annotations = self.df_annotations.append(df_annotations_modified[
															   ['dataset', 'subset', 'test_train_val', 'folder',
																'filename', 'path', 'label', 'width', 'height', 'x_min',
																'x_max', 'y_min', 'y_max', 'yolo_x', 'yolo_y', 'yolo_w',
																'yolo_h']],
														   ignore_index=True)

					logging.info("Saved annotations from {0}".format(os.path.join(base_folder, folder)))
					self.df_annotations.to_csv(self.annotations_dest_name, header=True, index=False)