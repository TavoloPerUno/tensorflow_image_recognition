from abc import ABC, abstractmethod
import sys
import os
import shutil
import pandas as pd

sys.path.append('..', )

from utils.gcloud_access import extract_files



class Annotations(ABC):
	def __init__(self, source):
		self.compressed_annotations_folder = os.path.join('..', 'data', 'training', 'annotations_compressed', source)
		self.annotations_folder = os.path.join('..', 'data', 'training', 'annotations', source)
		self.meta_folder = os.path.join('..', 'data', 'training', 'metadata')
		self.compressed_images_folder = os.path.join('..', 'data', 'training', 'images_compressed', source)
		self.images_folder = os.path.join('..', 'data', 'training', 'images', source)
		self.annotations_dest_name = os.path.join(self.meta_folder, 'gsv_annotations.csv')
		self.lst_compressed_annotations_file = []

		self.df_annotations = pd.DataFrame(
			columns=['dataset', 'subset', 'test_train_val', 'folder', 'filename', 'path', 'label', 'width', 'height',
					 'x_min', 'x_max', 'y_min', 'y_max', 'yolo_x', 'yolo_y', 'yolo_w', 'yolo_h'])

		if os.path.isfile(self.annotations_dest_name):
			self.df_annotations = pd.read_csv(self.annotations_dest_name)

	def download_data(self, is_compressed):
		if is_compressed:
			for annotation_file in self.lst_compressed_annotations_file:
				extract_files(
					os.path.join(self.compressed_annotations_folder, annotation_file),
					self.annotations_folder)
			shutil.rmtree(self.compressed_annotations_folder)
		return