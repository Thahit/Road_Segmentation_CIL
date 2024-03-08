import os
from PIL import Image

directory = "training/images"
directory_gt = "training/groundtruth"

with open ('train_list.txt', 'a') as f:
    for filename in os.listdir(directory):
        if str(filename).endswith(".png"):
            path = os.path.join(directory,filename)
            gt_path = os.path.join(directory_gt,filename)
            f.write(path +'|'+ gt_path + '\n')

