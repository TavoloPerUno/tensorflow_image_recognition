
import os
import logging
import zipfile

def extract_files(source_file, dest_folder):
	if not os.path.exists(dest_folder):
		os.makedirs(dest_folder)

	if zipfile.is_zipfile(source_file):
		zip_ref = zipfile.ZipFile(source_file, 'r')
		zip_ref.extractall(os.path.join(dest_folder, os.path.splitext(os.path.basename(os.path.normpath(source_file)))[0]))
		zip_ref.close()
		logging.info('Extraction from zip file complete')

# def download_from_bucket(bucket_name, path, projectid, dest_folder, mounted_folder=None, mounted=True):
# 	local_dest_name = os.path.join(dest_folder, os.path.basename(os.path.normpath(path))) if not mounted else os.path.join(mounted_folder, os.path.basename(os.path.normpath(path)))
#
# 	if not os.path.exists(dest_folder):
# 		os.makedirs(dest_folder)
#
# 	if not mounted:
#
# 		auth.authenticate_user()
# 		storage_client = storage.Client(project=projectid)
# 		bucket = storage_client.get_bucket(bucket_name)
# 		blob = bucket.blob(path)
#
# 		blob.download_to_filename(local_dest_name)
#
# 		logging.info('Download from google bucket complete')
#
# 	if zipfile.is_zipfile(local_dest_name):
# 		zip_ref = zipfile.ZipFile(local_dest_name, 'r')
# 		zip_ref.extractall(os.path.join(dest_folder, os.path.splitext(os.path.basename(os.path.normpath(path)))[0]))
# 		zip_ref.close()
#
# 		if not mounted:
# 			os.remove(local_dest_name)
# 		logging.info('Extraction from zip file complete')
#
# 	return

