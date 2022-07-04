import commonTools
import customizedYaml
import yaml
import pandas as pd
import regex as re
from PIL import Image
import multiprocessing
from functools import partial
from contextlib import contextmanager
import xml.etree.ElementTree as ET
import os


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def get_target(target):
    if bool(re.search('genus', target.lower())):
        return 8
    elif bool(re.search('species', target.lower())):
        return 9
    return 0

def get_core_slide(annotation_dir):
    annotated_folders = os.listdir(annotation_dir)
    search_names = []
    for a in annotated_folders:
        core, slide = commonTools.keyword_from_folder(a)
        search_name = core + "_" + slide
        search_name = search_name.lower()
        search_names.append(search_name)
    return search_names


def getting_all_labels(record_file, target, annotation_dir):
    """
    Input the record file and target. Get a unique list of all names for dictionary
    """
    target_no = get_target(target)
    search_names = get_core_slide(annotation_dir)
    record_file['search name'] = record_file[0] + '_' + record_file[1]
    matched_records = record_file[record_file['search name'].str.lower().isin(search_names)]
    lost_records = record_file[~record_file['search name'].str.lower().isin(search_names)]
    lost_records.to_csv('lost_records.csv', index=False, header=None)
    if target_no == 8:
        return matched_records[8].value_counts().index.tolist()
    if target_no == 9:
        species = matched_records[8] + ' ' + matched_records[9]
        return species.value_counts().index.tolist()
    raise 'Invalid target, should be either genus or species'


def replace_xml_data(annotation, image):
    width, height = image.size
    size = annotation.find('size')
    width_data = size.find('width')
    height_data = size.find('height')
    width_data.text = f'{width}'
    height_data.text = f'{height}'
    return annotation


def xml_operation(target_folder, grid_path, annotation_path):
    operation_grid = os.path.join(grid_path, target_folder)
    operation_annotation = os.path.join(annotation_path, target_folder)
    for files in commonTools.files(operation_annotation):
        matching_grid = files.replace('.xml', '.tif')
        matching_grid_path = os.path.join(operation_grid, matching_grid)
        annotation_file = ET.parse(os.path.join(operation_annotation, files))
        img_file = Image.open(matching_grid_path)
        final_annotation = replace_xml_data(annotation_file, img_file)
        final_annotation.write(os.path.join(operation_annotation, files))


def fix_xml_file_size():
    """
    ad-hoc tool for fixing the file width and height problem of the xml annotation file
    """
    annotation_path = 'D:\Competetion_data\Ostracods_data\Pseudo_annotation\pascal_voc'
    grid_path = 'D:\Competetion_data\Ostracods_data\Grid_images'

    all_annotations = [folders for folders in commonTools.folders(annotation_path)]
    # processes = multiprocessing.cpu_count() - 1
    # with poolcontext(processes=processes) as pool:
    #     pool.map(partial(xml_operation, grid_path=grid_path, annotation_path=annotation_path), all_annotations)
    for f in all_annotations:
        xml_operation(f, grid_path, annotation_path)


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def convert_yolo(xml_annotation, class_dict):
    result = []
    image_size = xml_annotation.find('size')
    width = int(image_size.find('width').text)
    height = int(image_size.find('height').text)

    objects = xml_annotation.findall('object')
    for obj in objects:
        class_name = obj.find('name').text
        if len(class_dict) == 1:
            class_ref = 0
        else:
            class_ref = class_dict.index(class_name)
        bbox = [int(x.text) for x in obj.find("bndbox")]
        yolo_bbox = xml_to_yolo_bbox(bbox, width, height)
        bbox_string = " ".join([str(x) for x in yolo_bbox])
        result.append(f"{class_ref} {bbox_string}")
    return result

def output_yolo_format(output_folder, output_file, output_content):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    output_path = os.path.join(output_folder, output_file)
    with open(output_path, 'w', encoding='ascii') as f_out:
        f_out.write('\n'.join(output_content))

def convert_entrance(folder, source_dir, output_dir, class_dict):
    source_folder = os.path.join(source_dir, folder)
    output_folder = os.path.join(output_dir, folder)
    for files in commonTools.files(source_folder):
        xml_annotation = ET.parse(os.path.join(source_folder, files))
        file_result = convert_yolo(xml_annotation, class_dict)
        yolo_output_name = files.replace('.xml', '.txt')
        output_yolo_format(output_folder, yolo_output_name, file_result)


def convertion_preparation(pascal_dir, yolo_dir, class_dict):
    all_folders = [folder for folder in commonTools.folders(pascal_dir)]
    processes = multiprocessing.cpu_count() - 1
    with poolcontext(processes=processes) as pool:
        pool.map(partial(convert_entrance, source_dir = pascal_dir, output_dir = yolo_dir, class_dict = class_dict
                         ), all_folders)

def build_class_dict(record_file, target, annotation_dir):
    if target == 'pseudo':
        return [0]
    return getting_all_labels(record_file, target, annotation_dir)


if __name__ == '__main__':
    params = commonTools.parse_opt()
    yaml_data = customizedYaml.yaml_handler(params.yaml)
    base_dir = yaml_data.data['base_path']
    grid_dir = yaml_data.build_new_path('base_path', 'grid_images')
    target = params.target
    annotation_dir = os.path.join(base_dir, f'{target}_annotation')
    pascal_dir = os.path.join(annotation_dir, 'pascal_voc')
    yolo_dir = os.path.join(annotation_dir, 'yolo')
    master_df = pd.read_csv(os.path.join(base_dir, 'all_records.csv'), header=None)
    classes = build_class_dict(master_df, target, pascal_dir)
    convertion_preparation(pascal_dir,yolo_dir,classes)
    classes_file = os.path.join(base_dir, f'{target}_classes.txt')
    with open(classes_file, 'w', encoding='ascii') as f_out:
        f_out.write('\n'.join(classes))
    # if mode == 1 convert pascal to yolo
    # if mode == 2 convert yolo to pascal
