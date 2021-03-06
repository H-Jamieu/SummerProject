# from DummyImageGenerator import GenerateDummy
import random
import os
import csv
import commonTools
import yaml
import customizedYaml


def CreateTrays(len, n):
    """
    Create trays for inference
    :param len: length of input array
    :param n: number of trays to be created
    :return: List of n trays
    """
    tray_list = []
    tray_0 = []
    for i in range(0, len):
        tray_0.append(i)
    tray_list.append(tray_0)
    for j in range(1, n):
        random.seed(j)
        tray_x = tray_0.copy()
        random.shuffle(tray_x)
        tray_list.append(tray_x)
    return tray_list


# if __name__ == '__main__':
#     image_path = '../Plaindata/'
#     phases = ['train', 'val', 'test']
#     for phase in phases:
#         ref_path = '../Metadata/Species_' + phase + '.csv'
#         out_labels = '../YoloImages/Labels/' + phase + '/'
#         out_images = '../YoloImages/Images/' + phase + '/'
#         imgs = []
#         classes = []
#         batch: int = 9
#         with open(ref_path, 'r', encoding='ascii', errors='ignore') as f_in:
#             csv_reader = csv.reader(f_in, delimiter=',')
#             for row in csv_reader:
#                 imgs.append(os.path.join(image_path, row[0]))
#                 classes.append(row[1])
#         f_in.close()
#         ctr = len(classes)  # controlling factor for debug limiting how may images are cropped
#         trays = CreateTrays(ctr, batch)
#         for t in range(0, ctr):
#             img_batch = []
#             class_batch = []
#             for q in range(0, batch):
#                 img_batch.append(imgs[trays[q][t]])
#                 class_batch.append(classes[trays[q][t]])
#             yolo_img, yolo_roi = GenerateDummy(img_batch)
#             yolo_img.save(out_images+'yolo_' + phase + '_' + str(t) + '.jpg', quality=100)
#             with open(out_labels + 'yolo_' + phase + '_' + str(t) + '.txt', 'w', encoding='ascii', errors='ignore') as f_out:
#                 for roi, cls in zip(yolo_roi, class_batch):
#                     roi_str = str(cls)+' '+' '.join(str(r) for r in roi)
#                     f_out.write(roi_str)
#                     f_out.write('\n')
#             f_out.close()

def read_classes(classes_file):
    with open(classes_file, 'r', encoding='ascii') as f_in:
        lines = f_in.readlines()
    f_in.close()
    return lines


def txt_tif(fn):
    return fn.replace('.txt', '.tif')


def generate_file_list(source_dir, grid_dir):
    """
    the source dir is recommended to be root/genus_annotation/yolo/*
    """
    yolo_files = []
    for folder in commonTools.folders(source_dir):
        source_folder = os.path.join(source_dir, folder)
        grid_folder = os.path.join(grid_dir, folder)
        all_files = [os.path.join(grid_folder, txt_tif(a)) for a in os.listdir(source_folder)]
        yolo_files += all_files
    return yolo_files


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}grid_images{os.sep}', f'{os.sep}genus_annotation{os.sep}yolo{os.sep}'
    # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def train_test_split(file_list, record, target, class_dict):
    """
    80% train, 10% validation, 10% test
    the splitting is based on the count of classes. Objects with at least 10 image would be used for training.
    customized train-test split: count how many objects are there in each annotation and balance between the object count
    """
    for f in file_list:
        core_name, slide_name, grid_no = commonTools.keyword_from_path(f)
        matched_grid = record[(record[0] == core_name) & (record[1] == slide_name) &
                              (record[2] == grid_no)]
        tag = matched_grid[commonTools.get_target(target)].iloc[0]
        tag_num = class_dict.index(tag)


def yolo_prepare(source_dir, base_dir, target, grid_dir):
    yolo_class_dir = f'{target}_classes.txt'
    f = open(os.path.join(base_dir, yolo_class_dir), 'r', encoding='ascii')
    all_classes = [a.replace('\n', '') for a in f.readlines()]
    f.close()
    n_classes = len(all_classes)
    file_list = generate_file_list(source_dir, grid_dir)
    out_path = os.path.join(base_dir, f'yolo_{target}_images.txt')
    yaml_path = os.path.join(base_dir, f'yolo_{target}.yaml')
    yaml_data = {'path': base_dir, 'train': f'autosplit_{target}_train.txt', 'test': f'autosplit_{target}_test.txt',
                 'val': f'autosplit_{target}_val.txt', 'nc': n_classes, 'names': all_classes}
    with open(out_path, 'w', encoding='ascii') as f_out:
        f_out.write('\n'.join(file_list))
    f_out.close()
    with open(yaml_path, 'w', encoding='ascii') as yaml_out:
        yaml.dump(yaml_data, yaml_out, default_flow_style=None)
    yaml_out.close()
    print(all_classes)
    print(n_classes)


if __name__ == '__main__':
    params = commonTools.parse_opt()
    yaml_data = customizedYaml.yaml_handler(params.yaml)
    grid_dir = yaml_data.build_new_path('base_path', 'grid_images')
    base_dir = yaml_data.data['base_path']
    target = params.target
    annotation_dir = os.path.join(base_dir, f'{target}_annotation')
    yolo_dir = os.path.join(annotation_dir, 'yolo')
    yolo_prepare(yolo_dir, base_dir, target, grid_dir)
