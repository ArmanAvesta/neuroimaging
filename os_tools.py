"""
nnCapsNet Project
Developed by Arman Avesta, MD
Aneja Lab | Yale School of Medicine
Created (11/1/2022)
Updated (11/1/2022)

This file contains functions to help with file & folder organization, parallel processing, and other operating system
functionalities.
"""

# ------------------------------------------------- ENVIRONMENT SETUP -------------------------------------------------

# Project imports:


# System imports:
import os
from os.path import join, dirname
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import torch
from typing import List


# Print configs:
np.set_printoptions(precision=1, suppress=True)
torch.set_printoptions(precision=1, sci_mode=False)



# ------------------------------------------------ HELPER FUNCTIONS ---------------------------------------------------



def subdirs(root: str, complete_path: bool = False, prefix: str = None, suffix: str = None, sort: bool = True) \
        -> List[str]:
    if complete_path:
        func = os.path.join
    else:
        func = lambda x, y: y
    res = [func(root, subdir) for subdir in os.listdir(root) if os.path.isdir(join(root, subdir))
           and (prefix is None or subdir.startswith(prefix))
           and (suffix is None or subdir.endswith(suffix))]
    if sort:
        res.sort()
    return res


def subfiles(root: str, complete_path: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) \
        -> List[str]:
    if complete_path:
        func = os.path.join
    else:
        func = lambda x, y: y
    res = [func(root, subdir) for subdir in os.listdir(root) if os.path.isfile(join(root, subdir))
           and (prefix is None or subdir.startswith(prefix))
           and (suffix is None or subdir.endswith(suffix))]
    if sort:
        res.sort()
    return res


def sub_niftis(root: str, complete_path: bool = True, sort: bool = True) -> List[str]:
    return subfiles(root, complete_path=complete_path, sort=sort, suffix='.nii.gz')



def split_path(path: str) -> List[str]:
    """
    splits at each separator. This is different from os.path.split which only splits at last separator
    """
    return path.split(os.sep)

# .....................................................................................................................

def log(content, log_path):
    """
    This function logs the string passed to it as content, and adds the string to the log file passed to it as
    log_path.

    :param content: {str}
    :param log_path: {str}
    :return: None. Side effect: adds content to the log file with log_path.
    """
    with open(log_path, 'a') as log_file:
        log_file.write(content)
        log_file.write(f'\n\n')


def print_and_log(content, log_path):
    """
    This function prints the string passed to it as content, and adds the string to the log file passed to it as
    log_path.

    :param content: {str}
    :param log_path: {str}
    :return: None. Side effects: prints content, and adds content to the log file with log_path.
    """
    print(content)
    with open(log_path, 'a') as log_file:
        log_file.write(content)
        log_file.write(f'\n\n')

# .....................................................................................................................

def change_strings_in_csvs(csv_files, old_string, new_string):
    """
    This function changes strings in CSV files.

    :param csv_files: {str} path to csv file
    :param old_string: {str} string to be replaced
    :param new_string: {str}
    :return: None. Side effects: saves updated csv file.
    """
    csv_files = [csv_files] if type(csv_files) is pd.DataFrame else csv_files
    for csv_file in csv_files:
        paths_df = pd.read_csv(csv_file)
        for i in range(len(paths_df)):
            paths_df.iloc[i, 0] = paths_df.iloc[i, 0].replace(old_string, new_string)
        paths_df.to_csv(csv_file)

# .....................................................................................................................

def recursive_rename(root, old_string, new_string):
    """
    This function replaces old_string in all file/folder under root. If a file/folder doesn't have old_string in their
    name, the function does not touch them.

    :param root: {str} path to root (under which files/folders should be changed)
    :param old_string: {str}
    :param new_string: {str}
    :return: None. Side effect: changed the names of all files/folders under root.
    """
    paths = list_filter_files_paths(root, old_string)
    for path in tqdm(paths, desc='renaming'):
        new_path = path.replace(old_string, new_string)
        os.system(f'mv {path} {new_path}')


def recursive_filter_and_copy(old_root, new_root, filename):
    """
    This function finds all files with "filename" in their path, then copies them from old root to new root while
    retaining the rest of the folder tree.

    :param old_root:
    :param new_root:
    :param filename:
    :return:
    """
    paths = list_filter_files_paths(old_root, filename)
    for path in tqdm(paths, desc='copying'):
        new_path = path.replace(old_root, new_root)
        new_folder = dirname(new_path)
        os.makedirs(new_folder, exist_ok=True)
        os.system(f'mv {path} {new_path}')


def recursive_delete(root, delete_phrases):
    delete_list = list_filter_files_paths(root, delete_phrases)
    print(delete_list)
    for path in tqdm(delete_list, desc='deleting'):
        os.system(f'rm -r {path}')

# .....................................................................................................................

def list_files_paths(root):
    """
    This function returns the paths to all files under the directory passed as root argument.

    :param root: {str} directory under which all files will be listed.
    :return: {list of strings} list of paths to all files under the directory.
    """
    files_paths = []
    for path, _, files in os.walk(root):
        for file in files:
            files_paths.append(join(path, file))
    return files_paths


def filter_files_paths(files_paths, filter_phrases):
    """
    This function filters a list of paths and returns only the list of paths that contain one of strings in the
    filter_phrases.

    :param files_paths: {list or tuple of strings, or a single string} a list of paths.
    :param filter_phrases: {list of tuple or strings} a list of filter phrases (strings). e.g. ['.nii.gz', 'T2.mgz'].
    :return: {list of strings} list of paths that contain one of filter phrases.
    """
    filter_phrases = [filter_phrases] if type(filter_phrases) is str else filter_phrases
    filtered_files_paths = []
    for filter_phrase in filter_phrases:
        for file_path in files_paths:
            if filter_phrase in file_path:
                filtered_files_paths.append(file_path)
    return filtered_files_paths


def negfilter_files_paths(files_paths, filter_phrases):
    """
    This function filters a list of paths and returns only the list of paths that contain one of strings in the
    filter_phrases.

    :param files_paths: {list or tuple of strings, or a single string} a list of paths.
    :param filter_phrases: {list of tuple or strings} a list of filter phrases (strings). e.g. ['.nii.gz', 'T2.mgz'].
    :return: {list of strings} list of paths that contain one of filter phrases.
    """
    filter_phrases = [filter_phrases] if type(filter_phrases) is str else filter_phrases
    filtered_files_paths = []
    for filter_phrase in filter_phrases:
        for file_path in files_paths:
            if filter_phrase not in file_path:
                filtered_files_paths.append(file_path)
    return filtered_files_paths


def list_filter_files_paths(root, filter_phrases):
    """
    This functions combines the two functions above and returns the paths to all files under the directory passed
    as root, and then filters and returns only the list of paths that contain one of the strings in the filter phrases.

    :param root: {str} directory under which all files will be listed.
    :param filter_phrases: {list of tuple or strings} a list of filter phrases (strings). e.g. ['.nii.gz', 'T2.mgz'].
    :return: {list of strings} list of paths that contain one of filter phrases.
    """
    filter_phrases = [filter_phrases] if type(filter_phrases) is str else filter_phrases
    files_paths = list_files_paths(root)
    filtered_files_paths = filter_files_paths(files_paths, filter_phrases)
    return filtered_files_paths


def list_filter_save_csv(root, filter_phrases, csv_path):
    """
    This function lists all files under the directory passed as root, filters the paths and only keeps the paths
    that contain one of filter_phrases, and saves the resulting list as a csv file (without any header or index).

    :param root: {str} directory under which all files will be listed.
    :param filter_phrases: {list of tuple or strings} a list of filter phrases (strings). e.g. ['.nii.gz', 'T2.mgz'].
    :param csv_path: {str} path of the csv file to be saved.
    :return:
    """
    filtered_files_paths = list_filter_files_paths(root, filter_phrases)
    filtered_files_paths_df = pd.DataFrame(filtered_files_paths)
    filtered_files_paths_df.to_csv(csv_path, header=False, index=False)

# .....................................................................................................................

def convert_csv_to_list(path_to_csv, header=None):
    """
    This function takes in the path to a csv file that contains paths to files, and returns a list of those paths.

    :param path_to_csv: {str} path to the csv file that contains paths of files. This csv file should have just
        one column, with no header and no indexes.
    :param header: whether the CSV has a header. If not, set to False.
    :return: {list of strings} list of paths to the files that are contained in the csv file.
    """
    df = pd.read_csv(path_to_csv, header=header)
    lst = list(df.iloc[:, 0])
    return lst


def convert_list_to_csv(lst, path_to_csv, header=False, index=False):
    """
    This function takes in a list and the path to a csv file, and saves a CSV file..

    :param list: {list}
    :param path_to_csv: {str} path to the csv file that contains paths of files. This csv file should have just
        one column, with no header and no indexes.
    :param header: whether the CSV should have a header. If not, set to False.
    :param index: whether the CSV should have indexes. if not, set to False.
    :return: None. Side effect: saves CSV.
    """
    df = pd.DataFrame(lst)
    df.to_csv(path_to_csv, header=header, index=index)


def save_csvs_subset(imgs_csv_path, segs_csv_path,  k, shuffle=True):
    """
    This function saves subsets of k images and k segmentations as csv files.

    :param imgs_csv_path: {str}
    :param segs_csv_path: {str}
    :param k: number of entries in the subset
    :return: None. Side-effects: saved csvs of images and segmentations with k entries.
    """
    imgs_list, segs_list = convert_csv_to_list(imgs_csv_path), convert_csv_to_list(segs_csv_path)
    assert len(imgs_list) == len(segs_list)

    indexes = list(range(len(imgs_list)))
    if shuffle:
        np.random.shuffle(indexes)
    subset_indexes = indexes[:k]

    imgs_subset_list = [imgs_list[index] for index in subset_indexes]
    segs_subset_list = [segs_list[index] for index in subset_indexes]

    imgs_subset_df, segs_subset_df = pd.DataFrame(imgs_subset_list), pd.DataFrame(segs_subset_list)
    imgs_sebset_path = imgs_csv_path.replace('.csv', f'_{k}.csv')
    segs_subset_path = segs_csv_path.replace('.csv', f'_{k}.csv')
    imgs_subset_df.to_csv(imgs_sebset_path, header=False, index=False)
    segs_subset_df.to_csv(segs_subset_path, header=False, index=False)

# .....................................................................................................................

def backup_to_s3(local_results_folder, s3_results_folder, verbose=False):
    """
    This method backs up the results to S3 bucket.
    It runs in the background and doesn't slow down training.
    """
    command = f'aws s3 sync {local_results_folder} {s3_results_folder}' if verbose \
        else f'aws s3 sync {local_results_folder} {s3_results_folder} >/dev/null &'

    os.system(command)
    print(f'>>>   S3 backup done   <<<')


# -------------------------------------------------- CODE TESTING -----------------------------------------------------

if __name__ == '__main__':

    images_folder = '/Users/sa936/projects/sccapsnet/data2/images'
    recursive_delete(images_folder, ['img_preprocessed.nii.gz', 'seg_preprocessed.nii.gz'])


