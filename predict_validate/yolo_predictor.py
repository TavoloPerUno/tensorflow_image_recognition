import os, sys
import time
import argparse
from skimage import io
import logging
import datetime
import csv
import cv2
import pandas as pd

# sys.path.append('/Users/Manu/Documents/pyWorkspace/darknet/python')
DARKNET_LOCATION = '/project2/kavibhalla/libraries/darknet'

sys.path.append(DARKNET_LOCATION)
sys.path.append('..', )

import predict_validate.darknet as dn
from predict_validate.predictor import Predictor

class YoloPredictor(Predictor):

	def __init__(self, model_name, data_folder, file_imagelist, file_class, images_from_api=False, images_location=None, file_class_grouping_json=None,
				 file_groundtruth=None, file_predictions=None):
		super().__init__(model_name, data_folder, file_imagelist, file_class, images_from_api, images_location, file_class_grouping_json,
						 file_groundtruth, file_predictions)

		self.cfg = os.path.join('..', 'custom_models', 'yolo', self.model_name, self.model_name + '.cfg')
		self.cdata = os.path.join('..', 'custom_models', 'yolo', self.model_name, self.model_name + '.data')

		self.weights = os.path.join('..', 'custom_models', 'yolo', self.model_name, self.model_name + '.weights')

		self.net = dn.load_net(self.cfg.encode('utf-8'), self.weights.encode('utf-8'), 0)
		self.meta = dn.load_meta(self.cdata.encode('utf-8'))

	def predict(self, val=False):

		df_predictions = pd.DataFrame(
			columns=['id', 'class', 'confidence', 'center_x', 'center_y', 'width', 'height', 'min_x', 'max_x', 'min_y',
					 'max_y'])

		if self.df_predictions.shape[0] > 0:
			df_predictions = self.df_predictions.copy()

		if not os.path.exists(os.path.join(self.data_folder, 'predictions')):
			os.makedirs(os.path.join(self.data_folder, 'predictions'))

		for idx, row in self.df_images[['id', 'location']].drop_duplicates().iterrows():
			logging.info("Processing image id " + str(row['id']))
			output_img = os.path.join(self.data_folder, 'predictions', str(row['id']) + '.jpg')
			while True:
				try:
					img = io.imread(row['location'] if self.images_from_api else os.path.join(self.images_location, str(row['id']) + '.jpg'))
					io.imsave(output_img, img)
					break
				except:
					logging.error("Error opening " + row['location'] if self.images_from_api else os.path.join(self.images_location, str(row['id']) + '.jpg'))

			pathb = output_img.encode('utf-8')
			res = dn.detect(self.net, self.meta, pathb)

			logging.info(res)  # list of name, probability, bounding box center x, center y, width, height
			img = cv2.imread(output_img)
			for i in range(len(res)):
				res_type = res[i][0].decode('utf-8')
				if res_type in self.lst_class:
					logging.info("Found a " + res_type)

					# get bounding box
					confidence = round(res[i][1], 2)
					center_x = int(res[i][2][0])
					center_y = int(res[i][2][1])
					width = int(res[i][2][2])
					height = int(res[i][2][3])

					UL_x = int(center_x - width / 2)  # Upper Left corner X coord
					UL_y = int(center_y + height / 2)  # Upper left Y
					LR_x = int(center_x + width / 2)
					LR_y = int(center_y - height / 2)

					# write bounding box to image
					cv2.rectangle(img, (UL_x, UL_y), (LR_x, LR_y), (255, 255, 255), 1)
					# put label on bounding box
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(img, res_type + '-' + str(confidence), (center_x, center_y), font, 0.3, (255, 255, 255),
								1, cv2.LINE_AA)
					# 	i = i + 1
					cv2.imwrite(os.path.join(self.data_folder, 'predictions', str(row['id']) + '.jpg'), img)

					df_predictions = df_predictions.append(pd.DataFrame({'id': row['id'],
																		 'class': res_type,
																		 'confidence': confidence,
																		 'center_x': center_x,
																		 'center_y': center_y,
																		 'width': width,
																		 'height': height,
																		 'min_x': UL_x,
																		 'max_x': LR_x,
																		 'min_y': LR_y,
																		 'max_y': UL_y
																		 },
																		index=[0]
																		, ),
														   ignore_index=True
														   )

			df_predictions.to_csv(os.path.join(self.data_folder, 'predictions.csv'), index=False)
			self.df_predictions = df_predictions

			if val:
				self.validate()


