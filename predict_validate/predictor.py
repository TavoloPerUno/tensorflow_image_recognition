from abc import ABC, abstractmethod
import logging
import pandas as pd
import json
import os
import csv

from predict_validate.helper import invert_class_grouping_dict

class Predictor(ABC):

	def __init__(self, model_name, data_folder, file_imagelist, file_class, images_from_api=False, images_location=None, file_class_grouping_json=None, file_groundtruth=None, file_predictions=None):
		self.model_name = model_name
		self.data_folder = data_folder
		self.df_images = pd.read_csv(file_imagelist)
		with open(file_class, 'r') as f:
			reader = csv.reader(f)
			self.lst_class = list(reader)[0]

		self.images_from_api = images_from_api
		self.images_location = images_location
		self.df_predictions = pd.DataFrame()
		self.df_groundtruth = pd.DataFrame()
		self.dct_class_grouping = dict()

		if not 	os.path.exists(data_folder):
			os.makedirs(data_folder)

		if file_groundtruth is not None:
			self.df_groundtruth = pd.read_csv(file_groundtruth)


		if file_class_grouping_json is not None:
			with open(file_class_grouping_json) as f:
				self.dct_class_grouping = json.load(f)
			self.dct_class_name_map = invert_class_grouping_dict(self.dct_class_grouping)


		if file_predictions is not None:
			self.df_predictions = pd.read_csv(file_predictions)

		super().__init__()

	def validate(self):
		if self.df_groundtruth.shape[0] > 0 and self.df_predictions.shape[0] > 0:
			df_prediction_agg = self.df_predictions.copy()
			df_prediction_agg['class'] = df_prediction_agg['class'].apply(lambda x: self.dct_class_name_map[x])

			if 'count' not in list(df_prediction_agg.columns):
				df_prediction_agg['is_included'] = df_prediction_agg.apply(lambda x: float(self.dct_class_grouping[x['class']]['thres']) <= x['confidence'], axis=1)
				df_prediction_agg = df_prediction_agg.loc[df_prediction_agg['is_included'],].groupby(['id', 'class'])['confidence'].count().reset_index(name='count')

			df_results = pd.DataFrame(columns=['id', 'class', 'tp', 'fp', 'tn', 'fn'])

			df_groundtruth_matched = self.df_groundtruth.loc[self.df_groundtruth.id.isin(df_prediction_agg.id),].copy()
			df_groundtruth_matched['class'] = df_groundtruth_matched['class'].apply(lambda x: self.dct_class_name_map[x])

			logging.info("Size of validation set: {0}".format(str(df_groundtruth_matched.shape[0])))

			if df_groundtruth_matched.shape[0] < 1:
				logging.error("No groundtruth file found")
				return 0

			for id in df_groundtruth_matched.id.unique():
				df_candidate_predictions = df_prediction_agg.loc[df_prediction_agg['id'] == id,]
				df_candidate_groundtruth = df_groundtruth_matched.loc[df_groundtruth_matched['id'] == id,]


				for category in self.dct_class_grouping:

					tp = fp = tn = fn = 0
					num_predictions = df_candidate_predictions.loc[df_candidate_predictions['class'].isin([category] + ([] if category not in self.dct_class_grouping else self.dct_class_grouping[category]['classes'])),]['count'].sum()
					num_truth = df_candidate_groundtruth.loc[df_candidate_groundtruth['class'].isin([category] + (
						[] if category not in self.dct_class_grouping else self.dct_class_grouping[
							category]['classes'])),]['count'].sum()

					if num_predictions == 0 or num_truth == 0:
						if num_truth > 0:
							fn = 1 if (num_truth == 1) else (3 if (num_truth == 2) else 6)
						elif num_predictions > 0:
							fp = num_predictions
						else:
							tn = 1

					elif num_truth == 1:
						tp = min(num_predictions, 3)
						fp = max(0, num_predictions - 3)
					elif num_truth == 2:
						tp = min(num_predictions, 5)
						fp = max(0, num_predictions - 5)
						fn = max(3 - num_predictions, 0)
					else:
						tp = num_predictions
						fn = max(6 - num_predictions, 0)

					#
					# elif num_predictions > num_truth:
					#
					# 	fp = num_predictions - num_truth
					# 	tp = num_truth
					#
					# else:
					# 	tp = num_predictions
					# 	fn = num_truth - num_predictions

					df_results = df_results.append(pd.DataFrame({'id': id,
													             'class': category,
													             'tp': tp,
													             'fp': fp,
													             'tn': tn,
													             'fn': fn
													            },
												                index=[0]
																),
												   ignore_index=True
												   )

			df_results = df_results.groupby('class')[['tp', 'fp', 'tn', 'fn']].sum().reset_index()
			df_results['sensitivity'] = df_results['tp'] / (df_results['tp'] + df_results['fn'])
			df_results['specificity'] = df_results['tn'] / (df_results['tn'] + df_results['fp'])
			df_results['precision'] = df_results['tp'] / (df_results['tp'] + df_results['fp'])
			df_results['npv'] = df_results['tn'] / (df_results['tn'] + df_results['fn'])
			df_results['miss_rate'] = df_results['fn'] / (df_results['fn'] + df_results['tp'])
			df_results['fall_out'] = df_results['fp'] / (df_results['fp'] + df_results['tn'])
			df_results['fdr'] = df_results['fp'] / (df_results['fp'] + df_results['tp'])
			df_results['for'] = df_results['fn'] / (df_results['fn'] + df_results['tn'])
			df_results['acc'] = (df_results['tp'] + df_results['tn']) / (df_results['tp'] + df_results['tn'] + df_results['fp'] + df_results['fn'])
			df_results['conf'] = df_results['class'].apply(lambda x: self.dct_class_grouping[x]['thres'])

			df_results.to_csv(os.path.join(self.data_folder, 'scores.csv'), index=False)

			return 1
		else:
			logging.error("No groundtruth file found")
			return 0

	@abstractmethod
	def predict(self):
		pass
