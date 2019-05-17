import os
import sys
import logging
import datetime
import argparse


sys.path.append('..', )

from annotation_normalisation.cityscapes import Cityscapes
from annotation_normalisation.miotcd import Miotcd
from annotation_normalisation.cbcl import Cbcl
from annotation_normalisation.mapillaryvistas import MapillaryVistas


logname = os.path.join('..', 'logs', 'tensorflow_image_recognition_annotation_normalisation {:%Y-%m-%d %H:%M:%S}.log'.format(datetime.datetime.now()))
handler = logging.FileHandler(logname, mode='a')

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logging = logging.getLogger()
logging.addHandler(handler)



def main(argv):
	parser = argparse.ArgumentParser(description='Convert annotation format')

	parser.add_argument('s', type=str,
						help='Dataset name. Supported types: cityscapes, mapillaryvistas')

	parser.add_argument('-t', type=str,
						help='annotation_type')

	parser.add_argument('-o', type=str,
						help='output_folder')

	parser.add_argument('-n', type=str,
						help='ignored classes')


	args = parser.parse_args()

	source = args.s

	if source == 'cityscapes':
		annotations = Cityscapes()

	elif source == 'miotcd':
		annotations = Miotcd()

	elif source == 'cbcl':
		annotations = Cbcl()

	else:
		annotations = MapillaryVistas()

	#download_data(source, is_compressed)
	annotations.normalise_annotations()


if __name__ == '__main__':
	main(sys.argv[1:])
