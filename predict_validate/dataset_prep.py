import urllib.parse
import pandas as pd
import numpy as np

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