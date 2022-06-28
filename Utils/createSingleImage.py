import cv2
import commonTools
import customizedYaml


def copy_selected_area(img, bboxs):
    """
    Copy all binding boxes in the image.
    Assume the order of the binding box is kept during extraction
    """
    objects = []
    for bbox in bboxs:
        # xmin, ymin, xmax, ymax
        sub_obj = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        objects.append(sub_obj)
    return objects


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

    def show_bbox(self):
        print(f'Box started with ({self.x_min}, {self.y_min}) ended with ({self.x_max},{self.y_max}).')


def get_box_data(xml_annotation):
    """
    Get ostracods form the xml file
    """
    objects = xml_annotation.findall('object')
    ostracods = []
    for obj in objects:
        # object: have one binding box, have class name
        bbox = [int(x.text) for x in obj.find("bndbox")]
        class_name = obj.find("name")
        ostracod = ostracod(class_name, bbox)
        ostracods.append(ostracod)
    return ostracods