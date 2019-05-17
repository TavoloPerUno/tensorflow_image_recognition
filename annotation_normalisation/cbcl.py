import os
import sys
import pandas as pd
import logging
import collections
import numpy as np
from xmljson import parker
import xml.etree.ElementTree as ET

sys.path.append('..', )

from annotation_normalisation.annotations import Annotations
from annotation_normalisation.annotation_utils import *
class Cbcl(Annotations):

	def __init__(self):
		super().__init__('cbcl')

	def normalise_annotation_row(self, dfrow):
		dfrow['dataset'] = 'cbcl'
		dfrow['filename'] = dfrow['filename'].strip()
		img_width, img_height = get_image_dimensions(
			os.path.join(self.images_folder, 'Original', dfrow['filename']))

		dfrow['width'] = img_width
		dfrow['height'] = img_height
		lst_objects = []

		if 'object' in dfrow:
			if (type(dfrow['object']) in [collections.OrderedDict, dict]):
				dfrow['object'] = [dfrow['object']]
			if type(dfrow['object']) is list:
				for dct_object in dfrow['object']:
					if type(dct_object) in [collections.OrderedDict, dict]:
						if not dct_object['deleted'] and 'polygon' in dct_object:

							dctlst_pt = dct_object['polygon']['pt']

							if type(dct_object['polygon']['pt']) in [collections.OrderedDict, dict]:
								dctlst_pt = [dct_object['polygon']['pt']]

							lst_pt = []
							for dct_pt in dctlst_pt:
								if type(dct_pt) in [collections.OrderedDict, dict]:
									lst_pt.append([dct_pt['x'], dct_pt['y']])

							if len(lst_pt) > 0:
								lst_bounds = polygon_to_bounding_box(lst_pt)
								yolo_x, yolo_y, yolo_w, yolo_h = get_yolo_coordinates(lst_bounds[0], lst_bounds[1],
																					  lst_bounds[2], lst_bounds[3],
																					  img_width, img_height)
								lst_objects.append((dct_object['name'].strip(), lst_bounds[0], lst_bounds[1],
													lst_bounds[2], lst_bounds[3], yolo_x, yolo_y, yolo_w, yolo_h))

								continue

					logging.info("Strange annotation from {0}".format(dfrow['filename']))
					logging.info(dct_object)
		if len(lst_objects) < 1:
			logging.info("Strange annotation from {0}".format(dfrow['filename']))
			logging.info(dfrow['object'])
			lst_objects.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
		dfrow['objects'] = lst_objects

		return dfrow[['dataset', 'filename', 'width', 'height', 'objects']]

	def normalise_annotations(self):
		base_folder = os.path.join(self.annotations_folder, 'Anno_XML')
		lst_xml_files = [pos_xml for pos_xml in os.listdir(base_folder) if
						 pos_xml.endswith('.xml')]

		dct_annotations = dict()
		for xml_file in lst_xml_files:
			dct_annotations[xml_file] = parker.data(ET.parse(os.path.join(base_folder, xml_file)).getroot(),
													preserve_root=False)

		# with open(os.path.join(base_folder, xml_file), "r") as inputxml:
		# 	dct_annotations[xml_file] = parker.data(fromstring(inputxml), preserve_root=False)

		df_annotations_raw = pd.DataFrame(dct_annotations)
		df_annotations_raw = df_annotations_raw.T
		df_annotations_raw.reset_index(level=0, inplace=True)
		df_annotations_modified = df_annotations_raw.apply(self.normalise_annotation_row, axis=1)

		df_annotations_modified = df_annotations_modified.objects \
			.apply(pd.Series) \
			.merge(df_annotations_modified,
				   right_index=True,
				   left_index=True) \
			.drop(["objects"], axis=1) \
			.melt(id_vars=['dataset', 'filename', 'width', 'height'], value_name="object") \
			.drop("variable", axis=1) \
			.dropna()
		lst_new_columns = ['label', 'x_min', 'x_max', 'y_min', 'y_max', 'yolo_x', 'yolo_y', 'yolo_w', 'yolo_h']
		for n, col in enumerate(lst_new_columns):
			df_annotations_modified[col] = df_annotations_modified['object'].apply(lambda anno: anno[n])
		df_annotations_modified = df_annotations_modified.drop('object', axis=1)
		df_annotations_modified['subset'] = ''
		df_annotations_modified['test_train_val'] = 'train'
		df_annotations_modified['folder'] = 'Original'
		df_annotations_modified['path'] = df_annotations_modified['filename'].apply(
			lambda x: os.path.join('cbcl', 'Original', x))

		self.df_annotations = self.df_annotations.append(df_annotations_modified[
												   ['dataset', 'subset', 'test_train_val', 'folder', 'filename', 'path',
													'label', 'width', 'height', 'x_min', 'x_max', 'y_min', 'y_max',
													'yolo_x', 'yolo_y', 'yolo_w', 'yolo_h']],
											   ignore_index=True)

		logging.info("Saved annotations from cbcl")
		self.df_annotations.to_csv(self.annotations_dest_name, header=True, index=False)
