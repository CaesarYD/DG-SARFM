import os
import pickle
import random
import re
from collections import defaultdict

import cv2
from PIL import Image
from torch.utils.data import Dataset
from .datasets_for_mae import build_transform, build_transform2

label_int_dict = {
    'SOC': {'BMP2': 0, 'BTR70': 1, 'T72': 2, 'BTR_60': 3, '2S1': 4, 'BRDM_2': 5, 'D7': 6, 'T62': 7, 'ZIL131': 8,
            'ZSU_23_4': 9},
    'New_FUSAR': {'Bridges': 0, 'Cargo': 1, 'CoastalLands_island': 2, 'Fishing': 3, 'LandPatches': 4, 'OtherShip': 5,
                  'SeaClutterWaves': 6, 'SeaPatches': 7, 'StrongFalseAlarms': 8, 'Tanker': 9},
    'SAR-ACD':{'A220': 0, 'A330': 1, 'A320321': 2, 'ARJ21': 3, 'Boeing737': 4, 'Boeing787': 5},
    'EOC-Scene': {'BRDM_2': 0, 'T72': 1, 'ZSU_23_4': 2},
    'EOC-Depression': {'2S1': 0, 'BRDM_2': 1, 'T72': 2, 'ZSU_23_4': 3},
    'EOC-Configuration': {'BMP2': 0, 'BRDM_2': 1, 'BTR70': 2, 'T72': 3},
    'EOC-Version': {'BMP2': 0, 'BRDM_2': 1, 'BTR70': 2, 'T72': 3}
}

label_name_dict = {
    'SOC': {'BMP2': 'BMP2, a type of Infantry Fighting Vehicle',
            'BTR70': 'BTR70, a type of Armored Personnel Carrier',
            'T72': 'T72, a type of Tank',
            'BTR_60': 'BTR60, a type of Armored Personnel Carrier',
            '2S1': '2S1, a type of Self-propelled Artillery',
            'BRDM_2': 'BRDM2, a type of Amphibious Armored Scout Car',
            'D7': 'D7, a type of Bulldozer',
            'T62': 'T62, a type of Tank',
            'ZIL131': 'ZIL131, a type of Truck',
            'ZSU_23_4': 'ZSU234, a type of Self-propelled Anti-aircraft Gun'},
    'New_FUSAR': {'Bridges': 0, 'Cargo': 1, 'CoastalLands_island': 2, 'Fishing': 3, 'LandPatches': 4, 'OtherShip': 5,
                      'SeaClutterWaves': 6, 'SeaPatches': 7, 'StrongFalseAlarms': 8, 'Tanker': 9},
    'SAR-ACD':{'A220': 0, 'A330': 1, 'A320321': 2, 'ARJ21': 3, 'Boeing737': 4, 'Boeing787': 5},
    'EOC-Scene': {'BRDM_2': 0, 'T72': 1, 'ZSU_23_4': 2},
    'EOC-Depression': {'2S1': 0, 'BRDM_2': 1, 'T72': 2, 'ZSU_23_4': 3},
    'EOC-Configuration': {'BMP2': 0, 'BRDM_2': 1, 'BTR70': 2, 'T72': 3},
    'EOC-Version': {'BMP2': 0, 'BRDM_2': 1, 'BTR70': 2, 'T72': 3}
}


class Datum:
    def __init__(self, impath="", label=0, domain=0, classname=""):
        assert isinstance(impath, str)
        assert os.path.isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class Few_shot_dataset_test(Dataset):
    def __init__(self, root, dataset_name, transform):
        super(Few_shot_dataset_test, self).__init__()
        self.transform = transform
        self.root = root
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(self.root, self.dataset_name)
        self.label_int = label_int_dict[dataset_name]
        self.label_name = label_name_dict[dataset_name]

        test_dataset_dir = os.path.join(self.dataset_dir, 'TEST')
        test_dataset = self.read_data(self.label_int, self.label_name, test_dataset_dir)

        self.datas_list = test_dataset

    def read_data(self, label_int, label_name, image_dir):
        items = []

        for root, dirs, files in os.walk(image_dir):
            files = sorted(files)
            for file in files:
                if os.path.splitext(file)[1] == '.jpeg' or '.jpg' or '.png':
                    impath = os.path.join(root, file)
                    idx = re.split('[/\\\]', impath).index(self.dataset_name)
                    label = label_int[re.split('[/\\\]', impath)[idx + 2]]
                    classname = label_name[re.split('[/\\\]', impath)[idx + 2]]
                    item = Datum(impath=impath, label=label, classname=classname)
                    items.append(item)
        return items

    def __getitem__(self, idx):
        img_dir = self.datas_list[idx].impath
        img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
        img = Image.fromarray(img)
        label = self.datas_list[idx].label
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.datas_list)


class Few_shot_dataset_train(Few_shot_dataset_test):
    def __init__(self, root, dataset_name, num_shots, seed, transform):
        Few_shot_dataset_test.__init__(self, root, dataset_name, transform)
        self.seed = seed
        self.root = root
        self.num_shots = num_shots
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(self.root, self.dataset_name)
        self.spilt_few_shot_dir = os.path.join(self.dataset_dir, 'spilt_few_shot')
        os.makedirs(self.spilt_few_shot_dir, exist_ok=True)

        train_dataset_dir = os.path.join(self.dataset_dir, 'TRAIN')
        train_dataset = self.read_data(self.label_int, self.label_name, train_dataset_dir)

        self.split_few_shot_file = os.path.join(self.spilt_few_shot_dir, f'{self.num_shots}-shots-{self.seed}.pkl')
        if not os.path.exists(self.split_few_shot_file):
            few_shot_file = self.generate_few_shot_dataset(train_dataset)
            with open(self.split_few_shot_file, 'wb') as file:
                pickle.dump(few_shot_file, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.split_few_shot_file, 'rb') as file:
            self.datas_list = pickle.load(file)

        self.transform = transform

    def generate_few_shot_dataset(self, *datasources, repeat=False):
        if self.num_shots < 1:
            if len(datasources) == 1:
                return datasources[0]
            return datasources

        print(f"Creating a {self.num_shots}-shot dataset")

        output = []

        for data_source in datasources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            random.seed(self.seed)
            for label, items in tracker.items():
                if len(items) >= self.num_shots:
                    sampled_items = random.sample(items, self.num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=self.num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, items):
        output = defaultdict(list)

        for item in items:
            output[item.label].append(item)

        return output



class Percent_shot_dataset(Few_shot_dataset_test):
    def __init__(self, root, dataset_name, percent, seed, transform):
        super().__init__(root, dataset_name, transform)
        self.percent = percent
        self.seed = seed
        self.dataset_dir = os.path.join(self.root, self.dataset_name)
        self.spilt_few_shot_dir = os.path.join(self.dataset_dir, 'spilt_few_shot')
        os.makedirs(self.spilt_few_shot_dir, exist_ok=True)

        train_dataset_dir = os.path.join(self.dataset_dir, 'TRAIN')
        train_dataset = self.read_data(self.label_int, self.label_name, train_dataset_dir)

        self.split_percent_file = os.path.join(self.spilt_few_shot_dir, f'{self.percent}-nums-{self.seed}.pkl')
        if not os.path.exists(self.split_percent_file):
            few_shot_file = self.generate_percent_dataset(train_dataset)
            with open(self.split_percent_file, 'wb') as file:
                pickle.dump(few_shot_file, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.split_percent_file, 'rb') as file:
            self.datas_list = pickle.load(file)

        self.transform = transform

    def generate_percent_dataset(self, *datasources, repeat=False):
        if self.percent == 0.:
            if len(datasources) == 1:
                return datasources[0]
            return datasources

        print(f"Creating a {self.percent}-dataset")

        output = []

        for data_source in datasources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            random.seed(self.seed)
            for label, items in tracker.items():
                sampled_items = random.sample(items, int(self.percent * len(items)))
                # else:
                #     if repeat:
                #         sampled_items = random.choices(items, k=self.num_shots)
                #     else:
                #         sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output


    def split_dataset_by_label(self, items):
        output = defaultdict(list)

        for item in items:
            output[item.label].append(item)

        return output



def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if is_train:
        dataset = Few_shot_dataset_train(args.data_path, args.dataset_name, args.num_shots, args.seed, transform)

    else:
        dataset = Few_shot_dataset_test(args.data_path, args.dataset_name, transform)

    return dataset


def build_dataset_percent(is_train, args):
    transform = build_transform(is_train, args)
    if is_train:
        dataset = Percent_shot_dataset(args.data_path, args.dataset_name, args.percent, args.seed, transform)

    else:
        dataset = Few_shot_dataset_test(args.data_path, args.dataset_name, transform)

    return dataset


if __name__ == '__main__':
    path = r'D:\dateset\SAR\MSTAR_JPEG_E'
    dataset_soc_train = Few_shot_dataset_train(path, 'New_FUSAR', 5, 2, None)

    dataset_soc_test = Few_shot_dataset_test(path, 'New_FUSAR', None)
    print(len(dataset_soc_train))
    print(len(dataset_soc_test))
