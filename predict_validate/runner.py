import argparse
import os
import logging
import datetime
import sys

from predict_validate.dataset_prep import edit_image_urls
from predict_validate.yolo_predictor import YoloPredictor

logname = os.path.join('..', 'logs', 'tensorflow_image_recognition_detection {:%Y-%m-%d %H:%M:%S}.log'.format(datetime.datetime.now()))
handler = logging.FileHandler(logname, mode='a')
data_folder = os.path.join('..', 'data', 'prediction')

def main(argv):
	global data_folder

	parser = argparse.ArgumentParser(description='Predict/ Validate with pretrained models')

	parser.add_argument('model_type', type=str,
						help='model type (Eg. Yolo)')

	parser.add_argument('model_name', type=str,
						help='model name')

	parser.add_argument('file_imagelist', type=str,
						help='image list file')

	parser.add_argument('job_name', type=str,
						help='job name')

	parser.add_argument('-images_folder', type=str,
						help='Location of downloaded images')

	parser.add_argument('-file_class', type=str,
						help='List of applicable classes (csv file)')

	parser.add_argument('-file_class_grouping_json', type=str,
						help='json file denoting class groupings')

	parser.add_argument('-file_groundtruth', type=str,
						help='Groundtruth file')

	parser.add_argument('-file_predictions', type=str,
						help='Predictions file')

	parser.add_argument('-k1', type=str,
						help='google key')
	parser.add_argument('-k2', type=str,
						help='google key')
	parser.add_argument('-k3', type=str,
						help='google key')
	parser.add_argument('-k4', type=str,
						help='google key')

	parser.add_argument('-new_key', type=int, default=0,
						help='change image url keys')

	parser.add_argument('-p', type=int, default=1,
						help='Predict')

	parser.add_argument('-val', type=int, default=0,
						help='Validate')

	parser.add_argument('-online_images', type=int, default=0,
						help='Should images be fetched from Google API?')

	parser.add_argument('-prediction_threshold', type=float, default=0.15,
						help='Prediction threshold')

	args = parser.parse_args()

	data_folder = os.path.join(data_folder, args.job_name)

	if args.new_key:
		lst_key = [args.k1, args.k2, args.k3, args.k4]
		edit_image_urls(args.file_imagelist, lst_key)

	if args.model_type == 'yolo':
		model = YoloPredictor(args.model_name, data_folder, args.file_imagelist, args.file_class, args.online_images, args.images_folder, args.file_class_grouping_json,
				 args.file_groundtruth, args.file_predictions)

		if args.p:
			model.predict()

		if args.val:
			model.validate(args.prediction_threshold)

if __name__ == '__main__':
	main(sys.argv[1:])