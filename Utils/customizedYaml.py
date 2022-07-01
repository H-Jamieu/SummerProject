import os
import yaml

'''
A customized yaml phrasing module is created in this file.
Expected calling API:
params = 
'''


class yaml_handler:
    def __init__(self, yaml_path):
        self.path = yaml_path
        with open(yaml_path, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                self.data = data
            except yaml.YAMLError as exc:
                print(exc)
                exit(200)

    def get_data(self, keyword):
        return self.data[keyword]

    def build_new_path(self, keyword_root, keyword_tgt):
        return os.path.join(self.data[keyword_root], keyword_tgt)

    def build_default_paths(self):
        base_path = self.data['base_path']
        self.data['genus_annotation'] = os.path.join(base_path, 'genus_annotation')
        self.data['species_annotation'] = os.path.join(base_path, 'species_annotation')
        self.data['pseudo_annotation'] = os.path.join(base_path, 'pseudo_annotation')
        self.data['genus_pascal_voc'] = os.path.join(self.data['genus_annotation'], 'pascal_voc')
        self.data['genus_yolo'] = os.path.join(self.data['genus_annotation'], 'yolo')
        self.data['species_pascal_voc'] = os.path.join(self.data['species_annotation'], 'pascal_voc')
        self.data['species_yolo'] = os.path.join(self.data['species_annotation'], 'yolo')
        self.data['pseudo_pascal_voc'] = os.path.join(self.data['pseudo_annotation'], 'pascal_voc')
        self.data['pseudo_yolo'] = os.path.join(self.data['pseudo_annotation'], 'yolo')