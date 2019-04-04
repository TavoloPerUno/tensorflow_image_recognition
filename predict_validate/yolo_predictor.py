import os, sys
import time
import argparse
from skimage import io
import logging
import datetime

# sys.path.append('/Users/Manu/Documents/pyWorkspace/darknet/python')
sys.path.append('/Users/Manu/Documents/pyWorkspace/darknet')
sys.path.append('..', )

import addons.darknet as dn
import pdb
import shutil
import numpy as np
import cv2
import pandas as pd
import urllib.parse

logname = os.path.join('..', 'logs', 'tensorflow_image_recognition_yolo_detection {:%Y-%m-%d %H:%M:%S}.log'.format(datetime.datetime.now()))
handler = logging.FileHandler(logname, mode='a')

def modify_url_components(url, dct_param):
	url_parts = list(urllib.parse.urlparse(url))
	query = dict(urllib.parse.parse_qsl(url_parts[4]))
	query.update(dct_param)
	url_parts[4] = urllib.parse.urlencode(query)
	return urllib.parse.urlunparse(url_parts)

def edit_image_urls(input_file, lst_key):

	df_images = pd.read_excel(input_file,  sheet_name='all_cities')

	lstdf_images = []

	for chunk, df_images_sub in df_images.groupby(np.arange(df_images.shape[0]) // (df_images.shape[0]//len(lst_key))):
		df_images_sub['x'] = df_images_sub['x'].apply(modify_url_components, args=({'key': lst_key[chunk]},))
		lstdf_images.append(df_images_sub)

	pd.concat(lstdf_images).to_csv(input_file, index=False)


def get_predictions(input_file, net, meta):

	df_images = pd.read_csv(input_file)
	df_predictions = pd.DataFrame(columns=['id', 'class', 'center_x', 'center_y', 'width', 'height', 'min_x', 'max_x', 'min_y', 'max_y'])

	for idx, row in df_images.iterrows():
		while True:
			output_img = os.path.join('..', 'data', 'jobs', 'yolo_prediction', 'output', str(row['id']) + '.jpg')
			img = io.imread(row['x'])
			io.imsave(output_img, img)


			pathb = output_img.encode('utf-8')
			res = dn.detect(net, meta, pathb)
			logging.info("Processing image id " + str(row['id']))
			logging.info(res)  # list of name, probability, bounding box center x, center y, width, height
			i = 0
			while i < len(res):
				res_type = res[i][0].decode('utf-8')

				# get bounding box
				center_x = int(res[i][2][0])
				center_y = int(res[i][2][1])
				width = int(res[i][2][2])
				height = int(res[i][2][3])

				UL_x = int(center_x - width / 2)  # Upper Left corner X coord
				UL_y = int(center_y + height / 2)  # Upper left Y
				LR_x = int(center_x + width / 2)
				LR_y = int(center_y - height / 2)

					# write bounding box to image
					cv2.rectangle(img, (UL_x, UL_y), (LR_x, LR_y), box_color, 5)
					# put label on bounding box
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(img, res_type, (center_x, center_y), font, 2, box_color, 2, cv2.LINE_AA)
					i = i + 1
				cv2.imwrite(new_path, img)  # wait until all the objects are marked and then write out.
				# todo. This will end up being put in the last path that was found if there were multiple
				# it would be good to put it all the paths.
				os.remove(path)  # remove the original

def main(argv):
	parser = argparse.ArgumentParser(description='Predict with yolo_prediction models')

	# parser.add_argument('cfg', type=str,
	# 					help='Config file')
	#
	# parser.add_argument('cdata', type=int,
	# 					help='Config data file')
	#
	# parser.add_argument('w', type=str,
	# 					help='Weights')

	parser.add_argument('f', type=str,
						help='image list')

	parser.add_argument('-k1', type=str,
						help='google key')
	parser.add_argument('-k2', type=str,
						help='google key')
	parser.add_argument('-k3', type=str,
						help='google key')
	parser.add_argument('-k4', type=str,
						help='google key')

	parser.add_argument('-n', type=int,
						help='change image url keys')


	args = parser.parse_args()

	# cfg = args.cfg
	# cdata = args.cdata
	#
	# weights = args.w
	input_file = args.f
	lst_key = [args.k1, args.k2, args.k3, args.k4]
	change_key = args.n

	if change_key==1:
		edit_image_urls(input_file, lst_key)

	# net = dn.load_net(cfg.encode('utf-8'), weights.encode('utf-8'), 0)
	# meta = dn.load_meta(cdata.encode('utf-8'))
	#
	# get_predictions(input_file, net, meta)
if __name__ == '__main__':
	main(sys.argv[1:])