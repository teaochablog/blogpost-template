'''
Author:
Matthew Bilton

Description:
Simple python script which extracts html content and images
folder for easy bucket syncing.

Command-Line Usage:
python -m wp_friendly_notebook <export_name>.html public_address <output_folder> 
'''

import sys
import os
import re
import base64
import hashlib
import shutil
from bs4 import BeautifulSoup

IMG_PREFIX = 'nb_img_'

def identify_base64_img(img_soup):
    '''
    Identifies whether the img soup element has a base64 encoded src
    and extracts it if so.
    Then returns a tuple (or None):
    (b64 data bytes, generated filename)
    '''
    search_pattern = r'data:image/(?P<fmt>\S+);base64,(?P<data>\S+)'
    m = re.search(search_pattern, img_soup['src'], re.DOTALL)
    if m is None:
        return None
    
    match_dict = m.groupdict()
    filename = hashlib.sha256(match_dict["data"].encode('utf-8')).hexdigest()[:16]
    filename = f'{IMG_PREFIX}{filename}.{match_dict["fmt"]}'
    data = bytes(match_dict['data'], 'utf8')

    return (data, filename)


def make_wp_friendly_notebook(html_filename, public_address, output_dir):
    with open(html_filename, 'r') as html_file:
        soup = BeautifulSoup(html_file.read(), 'html.parser')

    html_dir = os.path.dirname(html_filename)
    base_filename = os.path.basename(html_filename)
    base_subdir = re.sub('\.html', '', base_filename)
    images_subdir = os.path.join(base_subdir, 'img')

    # Step 0:   Extract mathjax setup
    mathjax_config = soup.find("script", {"type" : "text/x-mathjax-config"})

    # Step 1:   Extract just the body, and change its tag type to div
    #           (We change it to div because the html will be pasted into wordpress)
    soup = soup.body
    soup.name = 'div'
    soup.append(mathjax_config)

    # Step 2:   Find all images in the document.
    #           If the src of the image is a data-string extract it.
    #           If the src is an asset-path, save the path.
    #           Update the doc so that both point to <public_address>
    data_images = []
    asset_images = []
    img_soups = soup.find_all('img')
    for img_soup in img_soups:
        img_data = identify_base64_img(img_soup)
        if not img_data is None:
            data_images.append(img_data)
            img_soup['src'] = f'{public_address}/{images_subdir}/{img_data[1]}'
        elif re.match(r'^/?assets', img_soup['src']):
            img_filename = re.sub(r'^/?assets/', '', img_soup['src'])
            asset_images.append(img_filename)
            img_soup['src'] = f'{public_address}/{images_subdir}/{img_filename}'

    # Finally:  Save the results to the output_dir. 
    
    # First create the directory structure
    save_dir = output_dir
    if not os.path.isdir(save_dir):
        print(f'Creating directory: {save_dir}')
        os.mkdir(save_dir)

    images_save_dir = os.path.join(save_dir, images_subdir)
    if not os.path.isdir(images_save_dir):
        print(f'Creating directory: {images_save_dir}')
        os.mkdir(images_save_dir)
    
    # Remove any previously generated images in the dist folder
    # so that we don't accumulate unwanted images.
    imgs_to_del = os.listdir(images_save_dir)
    for img in imgs_to_del:
        if re.match(IMG_PREFIX, img):
            rm_img_path = os.path.join(images_save_dir, img)
            print(f'Removing file: {rm_img_path}')
            os.remove(rm_img_path)

    # Then save all of the generated files
    save_html_filename = os.path.join(save_dir, base_filename)
    with open(save_html_filename, 'w') as html_outfile:
        print(f'Writing file: {save_html_filename}')
        html_outfile.write(soup.prettify())

    for img in data_images:
        img_filename = os.path.join(images_save_dir, img[1])
        with open(img_filename, 'wb') as img_file:
            print(f'Writing file: {img_filename}')
            img_file.write(base64.decodebytes(img[0]))
    
    assets_dir = os.path.join(html_dir, 'assets')
    for img_filename in asset_images:
        img_path = os.path.join(assets_dir, img_filename)
        print(f'Copying file: {img_filename}')
        shutil.copy2(img_path, images_save_dir)
        

    

if __name__ == "__main__":
    make_wp_friendly_notebook(sys.argv[1], sys.argv[2], sys.argv[3])