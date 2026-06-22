import datetime
import pickle
import os
import shutil
import json
import csv
import sys

def save_to_file(name: str, folder_name: str, use_auto_structure:bool=True, format: str = 'pickle', **kwargs) -> str:
    """Save all the data passed in pickle file
    Args:
        name (str): name of the file
        folder_name (str): Folder

    Returns:
        str: path of the file
    """
    if folder_name[-1] != '/':
        folder_name += '/'

    my_dict = dict()
    for key, values in kwargs.items():
        my_dict[str(key)] = values
    file_name = name  # set file name with date string
    if use_auto_structure:
        folder_name = r'{0}{1}'.format(
            folder_name, datetime.datetime.now().strftime("%Y_%m_%d/"))
        if not os.path.exists(folder_name):
            folder_name = os.path.join(folder_name, f"Acquisition_1")
            os.makedirs(folder_name)
        else:
            n = len(os.listdir(folder_name))
            folder_name = os.path.join(folder_name, f"Acquisition_{n+1}")
            os.makedirs(folder_name)
    elif  not os.path.exists(folder_name) : 
        current_dir=os.path.dirname(sys.argv[0])
        print("The folder does not exist")
        if input(f"Do you want to create it ? (y/n), if no the data will be saved in {current_dir} \n")=="y":
            os.makedirs(folder_name)
        else: 
            folder_name=current_dir
            print(f"The data will be saved in {current_dir}")    
        
    if format == "pickle":
        with open(os.path.join(folder_name, file_name) + ".pickle", 'wb') as file:
            pickle.dump(my_dict, file)
        print("Data saved in {0}".format(
            folder_name + "/" + file_name + ".pickle"))
        return folder_name, folder_name + "/" + file_name + ".pickle"

    if format == "json":
        with open(os.path.join(folder_name, file_name) + ".json", 'w') as file:
            json.dump(my_dict, file)
        print("Data saved in {0}".format(
            folder_name + "/" + file_name + ".json"))
        return folder_name, folder_name + "/" + file_name + ".json"

    if format == "txt":
        with open(os.path.join(folder_name, file_name) + ".txt", 'w') as file:
            for key in my_dict.keys():
                file.write(str(key) + " : " + str(my_dict[key]) + "\n")
        print("Data saved in {0}".format(
            folder_name + "/" + file_name + ".txt"))
        return folder_name, folder_name + "/" + file_name + ".txt"


def get_keywords_pickle_file(path: str) -> list[str]:
    """Print and return the keyword of a pickle file from is path

    Args:
        path (str): path of the pickle file

    Returns:
        list[str]: list of keywords
    """
    with (open(path, "rb") as fx):
        datax = pickle.load(fx)
    print("keyword : {0}".format(datax.keys()))
    return list(datax.keys())


def get_data_file(path: str, key=False, **kwargs):
    """Return the data from a pickle file.

    Args:
        path (str): Path to the pickle file.
        key (bool): If True, prints the available keys in the pickle file.
        **kwargs: Filters to select specific data by key.

    Returns:
        list: A list of data matching the specified filters.
    """
    _, extension = os.path.splitext(path)
    if extension == ".pickle":
        with open(path, "rb") as fx:
            datax = pickle.load(fx)
    if extension == ".json":
        with open(path, 'r') as openfile:
            datax = json.load(openfile)

    if extension == ".txt":
        with open(path, 'r') as openfile:
            datax = openfile.read()
            return datax
    if key:
        print("Keywords: {0}".format(datax.keys()))
    if len(kwargs.items()) == 0:
        # Return all data as a list
        return [datax[str(i)] for i in datax]
    else:
        key_dict = []
        for k, values in kwargs.items():
            if k in datax.keys():
                key_dict.append(k)
        return [datax[str(i)] for i in key_dict]

def update_file(path, concatenate: bool, **kwargs):
    _, extension = os.path.splitext(path)
    backup_path = path + ".bak"
    try:
        # Read existing data
        if extension == ".pickle":
            with open(path, "rb") as fx:
                datax = pickle.load(fx)
        elif extension == ".json":
            with open(path, "r") as openfile:
                datax = json.load(openfile)
        else:
            raise ValueError("Unsupported file format")
        shutil.copy(path, backup_path)
        keys = list(datax.keys())
        new_dict = datax.copy()  # Copy to avoid modifying original in case of failure
        for key, values in kwargs.items():
            if key in keys and concatenate:
                if isinstance(datax[str(key)], list):
                    newlist = datax[str(key)]
                elif isinstance(datax[str(key)], (str, float, int)):
                    newlist = [datax[str(key)]]
                else:
                    raise ValueError(f"Unsupported data type for key {key}")
                if isinstance(values, list):
                    newlist.extend(values)
                else:
                    newlist.append(values)
                new_dict[str(key)] = newlist
            else:
                new_dict[str(key)] = values
        temp_path = path + ".tmp"
        with open(temp_path, "wb" if extension == ".pickle" else "w") as openfile:
            if extension == ".pickle":
                pickle.dump(new_dict, openfile)
            else:
                json.dump(new_dict, openfile)
        os.replace(temp_path, path)
        os.remove(backup_path)
    except Exception as e:
        print(f"Error: {e}")
        print("file restore to previous version")
        if os.path.exists(backup_path):
            os.replace(backup_path, path)
        os.remove(temp_path)
                
def copy_file(Init_path, end_path, keyword):
    '''
    Copy every file containing the keyword to another folder
    :param Init_path: path of the initial folder
    :param end_path: path of the end folder
    :param keyword: keyword to find in the name of the file
    :return:
    '''
    folder_name = [f for f in os.listdir(Init_path)]
    for file in folder_name:
        if file.find(keyword) != -1:
            data_files = [f for f in os.listdir(Init_path + file)]
            shutil.copy(Init_path+file, end_path)


def find_non_utf8_files(directory: str) -> None:
    """Print all the Non-UTF-8 file ina directory

    Args:
        directory (str): path of the directory
    """
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read()
            except UnicodeDecodeError:
                print(f"Non-UTF-8 file: {file_path}")


def find_and_copy_utf8_files(source_dir, temp_dir, extension=".py"):
    """
    Finds files with wanted extension with UTF-8 encoding and copies them to a temporary directory.
    Skips problematic files.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                try:
                    # Check if the file is UTF-8 encoded
                    with open(file_path, "r", encoding="utf-8") as f:
                        f.read()
                        print(file_path)

                    # Copy the file to the temp directory
                    relative_path = os.path.relpath(root, source_dir)
                    dest_dir = os.path.join(temp_dir, relative_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(file_path, dest_dir)
                except UnicodeDecodeError:
                    print(f"Skipping non-UTF-8 file: {file_path}")


def get_main_parent_folder_path(n: int = 1):
    import sys
    dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    for i in range(n):
        dir = os.path.dirname(dir)
    return dir


def get_file_extension(file_path):
    """Returns the file extension of the given file path."""
    _, extension = os.path.splitext(file_path)
    return extension  # Includes the dot (e.g., '.py')

def get_file_parent_folder_path(path, n: int = 0):
    dir = os.path.abspath(os.path.dirname(path))
    for i in range(n):
        dir = os.path.dirname(dir)
    return dir