import os
from shutil import copy, rmtree
import random


def make_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

# ------------------------------------ step 1.2 : 训练,验证,测试集划分------------------------------------
def split_data():
    random.seed(15)
    split_rate = 0.1

    cwd = os.getcwd()
    data_root = os.path.join(cwd, "flower_classification/data")
    origin_flower_path = os.path.join(data_root, "flower_photos")
    assert os.path.exists(origin_flower_path), (f"path {origin_flower_path} does not exist")

    flower_class = [cla for cla in os.listdir(origin_flower_path) if os.path.isdir(os.path.join(origin_flower_path, cla))]

    train_root = os.path.join(data_root, "train")
    make_file(train_root)
    for cla in flower_class:
        make_file(os.path.join(train_root, cla))

    val_root = os.path.join(data_root, "val")
    make_file(val_root)
    for cla in flower_class:
        make_file(os.path.join(val_root, cla))
    
    for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        eval_index = random.sample(images, k=int(num*split_rate))
        print(eval_index)
        for index, image in enumerate(images):
            if image in eval_index:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            # print(f"\r[{cla}] processing [{index+1}/{num}]")
    
    print("processing done!")

if __name__ == "__main__":
    split_data()