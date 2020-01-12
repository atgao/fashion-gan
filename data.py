from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pathlib
import torch.utils.data as data
import torch
from config import *
import os
from PIL import Image
import random


class Fashion_attr_prediction(data.Dataset):
    def __init__(self, categories, type="train",
                 transform=None, target_transform=None, crop=False, img_path=None,
                 sample_size=10
                 ):
        self.transform = transform
        self.target_transform = target_transform
        self.crop = crop
        self.type = type
        self.train_list = []
        self.test_dict = {i: [] for i in categories}
        self.train_dict = {i: [] for i in categories}
        self.sample_dict = {i: [] for i in categories}
        self.test_list = []
        self.all_list = []
        self.sample_list = []
        self.sample_size = sample_size
        self.bbox = dict()
        self.anno = dict()
        self.read_partition_category()
        self.read_bbox()

    def __len__(self):
        if self.type == "all":
            return len(self.all_list)
        elif self.type == "train":
            return len(self.train_list)
        elif self.type == "test":
            return len(self.test_list)
        elif self.type == "sample":
            return len(self.sample_list)
        else:
            return 1

    def read_partition_category(self):
        list_eval_partition = os.path.join(DATASET_BASE, r'Eval', r'list_eval_partition.txt')
        list_category_img = os.path.join(DATASET_BASE, r'Anno', r'list_category_img.txt')
        partition_pairs = self.read_lines(list_eval_partition)
        category_img_pairs = self.read_lines(list_category_img)

        for img_path, category in category_img_pairs:
            category = int(category)
            if category in self.train_dict:
                self.anno[img_path] = category

        for img_path, partition in partition_pairs:
            if img_path not in self.anno:
                continue
            if partition == "train":
                self.train_list.append(img_path)
                self.train_dict[self.anno[img_path]].append(img_path)
            else:
                # Test and Val
                self.test_list.append(img_path)
                self.test_dict[self.anno[img_path]].append(img_path)

        self.all_list = self.test_list + self.train_list

        for category, sample_images in self.test_dict.items():
            sampled_test_images = random.choices(sample_images, k=self.sample_size)
            self.sample_dict[category] = sampled_test_images
            self.sample_list.extend(sampled_test_images)

        random.shuffle(self.train_list)
        random.shuffle(self.test_list)
        random.shuffle(self.all_list)
        random.shuffle(self.sample_list)

    def read_bbox(self):
        list_bbox = os.path.join(DATASET_BASE, r'Anno', r'list_bbox.txt')
        pairs = self.read_lines(list_bbox)
        for k, x1, y1, x2, y2 in pairs:
            self.bbox[k] = [int(x1), int(y1), int(x2), int(y2)]

    def read_lines(self, path):
        with open(path) as fin:
            lines = fin.readlines()[2:]
            lines = list(filter(lambda x: len(x) > 0, lines))
            pairs = list(map(lambda x: x.strip().split(), lines))
        return pairs

    def read_crop(self, img_path):
        img_full_path = os.path.join(DATASET_BASE, img_path)
        with open(img_full_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        if self.crop:
            x1, y1, x2, y2 = self.bbox[img_path]
            if x1 < x2 <= img.size[0] and y1 < y2 <= img.size[1]:
                img = img.crop((x1, y1, x2, y2))
        return img

    def __getitem__(self, index):
        if self.type == "all":
            img_path = self.all_list[index]
        elif self.type == "train":
            img_path = self.train_list[index]
        elif self.type == "sample":
            img_path = self.sample_list[index]
        else:
            img_path = self.test_list[index]
        target = self.anno[img_path]
        img = self.read_crop(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            print(self.target_transform)
            target = self.target_transform(target)
        category = self.anno[img_path]
        return img, category # if self.type == "all" else target


class GeneratedDataset(data.Dataset):
    def __init__(self, base_dir, transform=None, target_transform=None):
        """

        :param images: list of images
        :param transform:
        :param target_transform:
        """
        self.transform = transform
        self.target_transform = target_transform
        path = pathlib.Path(base_dir)
        # match all images in directory
        self.img_paths = [
            str(p) for p in
            list(path.glob('**/*.jpg')) + list(path.glob('**/*.png'))]

    def __len__(self):
        return len(self.img_paths)

    def read_img(self, img_path):
        with open(img_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        return img

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = self.read_img(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_path
