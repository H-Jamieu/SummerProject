import os.path

import commonTools
import customizedYaml
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image

import regex as re
from pathlib import Path

'''
This script is aiming at tagging the grid files using the cleaned record file.
Some problems we may face and handle in the program:
    1. Some photos are not taken but being recorded.
    2. Some photos are taken but not being recorded.

Creation process of the files are expected to have following properties:
    1. Returned file should be under PASCAL VOC format.
    2. Count of the actual ostracods vs the recorded amount should be recorded.
    3. Each grid contains only ONE species. If more than one found, send for error processing.
    4. Species = genus + ' ' + species
'''

def make_empty_xml(image_dir, sub_folder,image_file):
    # Core name example: HK14TLH1C_0_1_50X

    image_full_folder = os.path.join(image_dir, sub_folder)
    image_path = os.path.join(image_full_folder, image_file)

    if os.path.isfile(image_path):
        img = Image.open(image_path)
    else:
        print('Image not found: ', image_path)
        return
    width, height = img.size

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = os.path.basename(image_dir)
    ET.SubElement(annotation, 'filename').text = image_file
    ET.SubElement(annotation, 'path').text = image_path
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = f'{width}'
    ET.SubElement(size, 'height').text = f'{height}'
    ET.SubElement(size, 'depth').text = f'{8}'
    ET.SubElement(annotation, 'segmented').text = '0'

    tree = ET.ElementTree(annotation)
    return tree


def replace_xml_name(new_name, original_xml):
    """
    Replace the object names in the xml file to the tagged names in the PASCAL_VOC annotation file
    """
    objects = original_xml.findall('object')
    obj_cnt = 0
    for obj in objects:
        obj_name = obj.find('name')
        obj_name.text = new_name
        obj_cnt += 1
    return original_xml, obj_cnt


def get_target(target):
    if bool(re.search('genus', target.lower())):
        return 8
    elif bool(re.search('species', target.lower())):
        return 9
    return 0


def get_data_by_target(target, row):
    if target == 8:
        return row[8]
    if target == 9:
        return row[8] + ' ' + row[9]


def build_files(record_file, target):
    """
    This function is building the iconic parts of the files for finding the file names.
    pattern: HKUV12_FD_1_[\d,\dd]X_grid_1.tif

    run first, then efficiency
    if the grid is not recorded:
        if annoation files exists:
            replace('Ostracods', 'unidentified')
            if it is not empty:
                record error
        else:
            create empty annotation file
    if the grid is recorded but without annotation:
        record error

    """
    pattern_dict = {}
    target_index = get_target(target)
    if target_index == 0:
        raise ('Invalid target, should be genus or species')
    for idx, row in record_file.iterrows():
        folder_id_name = row[0] + '_' + row[1] + '_' + '[\d,\dd]X'
        grid_id_name = folder_id_name + '_grid_' + str(row[2])
        data_tuple = (grid_id_name, get_data_by_target(target_index, row))
        pattern_dict[folder_id_name] = data_tuple
    return pattern_dict


def create_annotation_name(record, grid_no, target):
    """
    This record should be a short-listed record with same core and slide and grid.
    """
    matched = record[record[2] == grid_no].iloc[0]
    if target == 8:
        fill_name = matched[8]
    else:
        fill_name = matched[8] + ' ' + matched[9]
    return fill_name


def process_full_info(full_info):
    if len(full_info) == 5 or len(full_info) == 6:
        info_data = {'core': full_info[0], 'grid_no': int(full_info[-1])}
        if len(full_info) == 5:
            info_data['slide'] = full_info[1]
        else:
            info_data['slide'] = full_info[1] + '_' + full_info[2]
        return info_data
    return 0


def label_annotations(folder, label_record, target, pseudo_dir, out_dir, image_path):
    """
    the folder here should be Absloute path for the annoation path.
    """
    if len(label_record.columns) == 10:
        # The last column left for stating for processed status
        label_record[10] = 0
    target_no = get_target(target)
    if target_no == 0:
        raise ('Invalid target, should be genus or species')
    pseudo_folder = os.path.join(pseudo_dir, folder)
    for annotations in commonTools.files(pseudo_folder):
        full_info = process_full_info(Path(annotations).stem.split('_'))
        if full_info == 0:
            print(annotations, ' not match the required format')
            continue
        # Test first. If successful, record the process.
        # A database would be extremely useful
        matched_grids = label_record[(label_record[0] == full_info['core']) & (label_record[1] == full_info['slide']) &
                                     (label_record[2] == full_info['grid_no'])]
        out_folder = os.path.join(out_dir, folder)
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
        if len(matched_grids) == 0:
            # create empty annotation
            image_file = annotations.replace('.xml', '.tif')
            empty_annotation = make_empty_xml(image_path, folder,image_file)
            if empty_annotation is None:
                continue
            empty_annotation.write(os.path.join(out_folder, annotations))
            print(annotations, ' not being recorded in the annotation file')
            continue
        annotation_name = create_annotation_name(matched_grids, full_info['grid_no'], target_no)
        xml_annotation = ET.parse(os.path.join(pseudo_folder, annotations))
        replace_xml_name(annotation_name, xml_annotation)
        xml_annotation.write(os.path.join(out_folder, annotations))

def replica(obj, rep):
    return [obj] * rep

def label_target(pseudo_path, label_record, target, out_dir, image_path):
    # parallel by folders
    # Abandon
    all_folders = []
    for folders in commonTools.folders(pseudo_path):
        label_annotations(folders, label_record, target, pseudo_path, out_dir, image_path)






if __name__ == '__main__':
    params = commonTools.parse_opt()
    yaml_data = customizedYaml.yaml_handler(params.yaml)
    base_dir = yaml_data.data['base_path']
    grid_dir = yaml_data.build_new_path('base_path', 'grid_images')
    yaml_data.data['genus_annotation'] = yaml_data.build_new_path('base_path', 'genus_annotation')
    yaml_data.data['pseudo_annotation'] = yaml_data.build_new_path('base_path', 'pseudo_annotation')
    pseudo_pascal_dir = yaml_data.build_new_path('pseudo_annotation', 'pascal_voc')
    genus_pascal_dir = yaml_data.build_new_path('genus_annotation', 'pascal_voc')
    target = 'genus'
    all_data = pd.read_csv(os.path.join(base_dir, 'all_records.csv'), header=None)
    label_target(pseudo_pascal_dir, all_data, target, genus_pascal_dir, grid_dir)
