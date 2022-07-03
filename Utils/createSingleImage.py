import os

import cv2
import commonTools
import customizedYaml
import xml.etree.ElementTree as ET
import multiprocessing
from functools import partial
from contextlib import contextmanager

"""
The script is creating the training set for classification. Ideally, we should output images under:
        $base_dir&\class_iamges\genus_species\Core_slide_grid_1_ind_1.tif
And output marking csv in:
        $base_dir$\class_labels\*
The labels should be:
        $path_to_img$, class_label (genus or species)
Multi-processing:
        Distribute cores based on the image folder under the root folder
"""
@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

class ostracod:
    def __init__(self, name, bbox):
        self.image = None
        self.grid_no = None
        self.slide_name = None
        self.core_name = None
        self.save_name = None
        self.class_name = name
        self.bbox = bbox
        self.x_min = bbox[0]
        self.y_min = bbox[1]
        self.x_max = bbox[2]
        self.y_max = bbox[3]

    def get_name(self):
        return self.class_name

    def get_bbox(self):
        return self.bbox

    def get_startpoint(self):
        return self.x_min, self.y_min

    def get_endpoint(self):
        return self.x_max, self.y_max

    def get_width_height(self):
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        return width, height

    def copy_selected_area(self, img):
        """
        Copy all binding boxes in the image.
        Assume the order of the binding box is kept during extraction
        """
        # xmin, ymin, xmax, ymax
        bbox = self.bbox
        sub_obj = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        self.image = sub_obj

    def show_bbox(self):
        print(f'Box started with ({self.x_min}, {self.y_min}) ended with ({self.x_max},{self.y_max}).')

    def store_orig_info(self, grid_dir):
        core_name, slide_name, grid_no = commonTools.keyword_from_path(grid_dir)
        self.save_name = core_name + '_' + slide_name + '_' + str(grid_no)
        self.core_name = core_name
        self.slide_name = slide_name
        self.grid_no = grid_no

    def box_error_checking(self):
        if self.image is None:
            print(f'{self.save_name} found non-exist images, please check')
            return False
        height, width, channels = self.image.shape
        if width<=0 or height<=0:
            print(f'{self.save_name} has invalid binding box, please fix the problem.')
            return False
        return True


def get_box_data(xml_annotation, grid_image):
    """
    Get ostracods form the xml file
    """
    objects = xml_annotation.findall('object')
    ostracods = []
    for obj in objects:
        # object: have one binding box, have class name
        bbox = [int(x.text) for x in obj.find("bndbox")]
        class_name = obj.find("name").text
        ostracod_ind = ostracod(class_name, bbox)
        ostracod_ind.copy_selected_area(grid_image)
        ostracods.append(ostracod_ind)
    return ostracods


def annotation2img(annotation_path):
    """
    Using pascal voc as default annotation and species as default label
    format:
    **/species_annotation/pascal_voc/** -> **/grid_images/**
    """
    img_path = annotation_path.replace('.xml', '.tif')
    return img_path.replace(f'species_annotation{os.sep}pascal_voc', 'grid_images')


def init_class_dir(image_path):
    if not os.path.isdir(image_path):
        os.mkdir(image_path)


def create_class_images(annotation_dir, class_dir):
    class_dir = 'E:\data\ostracods_id\class_images'
    xml_annotation = ET.parse(annotation_dir)
    grid_dir = annotation2img(annotation_dir)
    if not os.path.isfile(grid_dir):
        print(f'{grid_dir} does not exist, please check!')
        return
    grid_image = cv2.imread(grid_dir)
    ostracods_in_images = get_box_data(xml_annotation, grid_image)
    ind_cnt = 1
    for ostracod in ostracods_in_images:
        ostracod_folder = os.path.join(class_dir, ostracod.class_name)
        init_class_dir(ostracod_folder)
        ostracod.store_orig_info(grid_dir)
        if ostracod.box_error_checking():
            file_name = f'{ostracod.save_name}_ind{str(ind_cnt)}.tif'
            ostracod_path = os.path.join(ostracod_folder, file_name)
            cv2.imwrite(ostracod_path, ostracod.image)
            ind_cnt += 1


def make_class_files(annotation_base_dir, class_dir):
    annotation_base_dir = 'E:\data\ostracods_id\species_annotation\pascal_voc'
    all_annotations = []
    for folder in commonTools.folders(annotation_base_dir):
        annotation_folder = os.path.join(annotation_base_dir, folder)
        sub_annotations = [os.path.join(annotation_folder, a) for a in os.listdir(annotation_folder)]
        all_annotations += sub_annotations
    processes = multiprocessing.cpu_count() - 1
    with poolcontext(processes=processes) as pool:
        pool.map(partial(create_class_images, class_dir=class_dir), all_annotations)

if __name__=='__main__':
    params = commonTools.parse_opt()
    yaml_data = customizedYaml.yaml_handler(params.yaml)
    yaml_data.build_default_paths()
    pascal_annotation_path = yaml_data.data['species_pascal_voc']
    class_dir = yaml_data.build_new_path('base_path','class_images')
    make_class_files(pascal_annotation_path, class_dir)
