import os

path = "./results"


def get_fnames_recursive(dir_name):
    dir_list = os.listdir(dir_name)
    cur_fnames = []

    for item_name in dir_list:
        # if file
        if ".csv" in item_name:
            cur_fnames += [dir_name + "/" + item_name]
        # if folder
        elif "." not in item_name:
            sub_fnames = get_fnames_recursive(dir_name + "/" + item_name)
            cur_fnames += sub_fnames
    return cur_fnames

get_fnames_recursive(path)