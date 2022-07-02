import os

import cv2
import commonTools
import customizedYaml
import xml.etree.ElementTree as ET

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

class ostracod:
    def __init__(self, name, bbox):
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

    def append_img(self, img):
        self.img = img

    def show_bbox(self):
        print(f'Box started with ({self.x_min}, {self.y_min}) ended with ({self.x_max},{self.y_max}).')



def get_box_data(xml_annotation, grid_image):
    """
    Get ostracods form the xml file
    """
    objects = xml_annotation.findall('object')
    ostracods = []
    for obj in objects:
        # object: have one binding box, have class name
        bbox = [int(x.text) for x in obj.find("bndbox")]
        class_name = obj.find("name")
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
    img_path = annotation_path.replace('.xml','.tif')
    return img_path.replace(f'species_annotation{os.sep}pascal_voc', 'grid_images')

def create_class_images(grid_dir, annotation_dir):
    xml_annotation = ET.parse(annotation_dir)
    grid_image = cv2.imread(grid_dir)

