import pandas as pd
import commonTools
import customizedYaml
import os
import shutil

'''
The file is created for getting the files that require attention for re-tagging. An error csv file is provided for
reference. Out put of this script should be a folder holding files pending for fixing.

This script should also has the ability to:
    1. Record the fixed and un-fixed files, pack the files pending for repairing.
    2. Move the fixed files back to the dataset and replace the ones containing error
    3. update the fixed files in the error recording.

Some requirements for the design:
    a. Separate I/O and CPU work into different functions.
    b. Avoid using static variable. All static variables should be read from yaml file.
'''


def build_folder_name(core, slide):
    return core + '_' + slide


def move_to_error_bucket(source, dest):
    """
    Copy the image file under certain folder into a destination
    source: source path should be the EXACT path of the original file
    dest: destination file path could be a path, will be created if not exist.
    """
    if not os.path.exists(dest):
        os.mkdir(dest)
    if not os.path.exists(source):
        return
    shutil.copy2(source, dest)


def test_file_existance(source_path, annotation_path, dest_path):
    candidate_mags = ['40X', '80X', '100X']
    if os.path.isfile(source_path):
        return dest_path, annotation_path, source_path
    for cand in candidate_mags:
        new_source = source_path.replace('50X', cand)
        new_dest = dest_path.replace('50X', cand)
        new_annotation = annotation_path.replace('50X', cand)
        if os.path.isfile(new_source):
            return new_source, new_annotation, new_dest
    return source_path, annotation_path, dest_path


def getting_error(error_out_path, grid_path, error_info):
    # Possible magnify for files: 50X, 40X, 80X, 100X
    if 'processed' not in error_info:
        error_info['processed'] = 0
    for idx, row in error_info.iterrows():
        core = row['core']
        slide = row['slide']
        grid = row['grid']
        root_name = build_folder_name(core, slide)
        source_folder_name = build_folder_name(root_name, '50X')
        # Find $base_dir/grid_images/HK14TLH1C_0_1_50X/HK14TLH1C_0_1_50X_grid_1.tif
        # for folder in commonTools.conditional_folders(grid_path, target_folder_name):
        #     # Folder should be: HK14TLH1C_0_1_50X
        #     grid_file_path = os.path.join(grid_path, folder)
        #     magnify = grid_file_path.split('_')[-1]
        #     source_path = grid_file_path
        source_grid = root_name + '_50X' + '_grid_' + str(grid) + '.tif'
        source_annoatation = root_name + '_50X' + '_grid_' + str(grid) + '.xml'
        # $base_dir/errors/HK14TLH1C_0_1_50X
        dest_folder_path = os.path.join(error_out_path, source_folder_name)
        # $base_dir/grid_images/HK14TLH1C_0_1_50X
        source_folder_path = os.path.join(grid_path, source_folder_name)
        # $base_dir / grid_images / HK14TLH1C_0_1_50X / HK14TLH1C_0_1_50X_grid_1.tif
        final_source_path = os.path.join(source_folder_path, source_grid)
        final_annotation_path = os.path.join(source_folder_path, source_annoatation)
        final_source_path, final_annotation_path, dest_folder_path \
            = test_file_existance(final_source_path, final_annotation_path, dest_folder_path)
        if os.path.isfile(final_source_path):
            move_to_error_bucket(final_source_path, dest_folder_path)
            move_to_error_bucket(final_annotation_path, dest_folder_path)
            error_info.loc[idx, 'processed'] = 1
    return error_info


def remove_error_files(full_record, error_record):
    full_record[2] = full_record[2].astype(str)
    error_record[2] = error_record[2].astype(str)
    full_record['search name'] = full_record[0] + '_' + full_record[1] + '_grid_' + full_record[2]
    error_record['search name'] = error_record[0] + '_' + error_record[1] + '_grid_' + error_record[2]
    error_files = error_record['search name'].tolist()
    error_free = full_record[~full_record['search name'].isin(error_files)]
    return error_free


if __name__ == '__main__':
    '''
    All IO should be handled in main or specially designed IO functions
    '''
    params = commonTools.parse_opt()
    yaml_data = customizedYaml.yaml_handler(params.yaml)
    base_path = yaml_data.get_data('base_path')
    error_path = yaml_data.build_new_path('base_path', 'error_record.csv')
    grid_path = yaml_data.build_new_path('base_path', 'grid_images')
    error_info = pd.read_csv(error_path, header=None)
    if params.mode == 0:
        error_out_path = yaml_data.build_new_path('base_path', 'errors')
        error_info = getting_error(error_out_path, grid_path, error_info)
        error_info.to_csv(error_path, index=False)
    if params.mode == 1:
        all_records = pd.read_csv(os.path.join(base_path, 'all_records_raw.csv'), header=None)
        error_free = remove_error_files(all_records, error_info)
        error_free_out_path = os.path.join(base_path, 'all_records.csv')
        error_free.to_csv(error_free_out_path, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], header=None, index=False)
