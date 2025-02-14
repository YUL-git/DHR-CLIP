import torch.utils.data as data
import json
from PIL import Image
import numpy as np
import os
from utils_anyres import resize_and_patch_image

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, args, patch_transform=None, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.resize_resolution = (args.target_image_size, args.target_image_size)
        self.patch_size = args.patch_image_size
        self.patch_transform = patch_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                                data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        
        # transforms
        img, patch_imgs = resize_and_patch_image(img, resize_resolution=self.resize_resolution, patch_size=self.patch_size)

        img = self.transform(img) if self.transform is not None else img
        patch_imgs = [self.patch_transform(patch_img) if self.patch_transform is not None else patch_img for patch_img in patch_imgs]

        img_mask = self.target_transform(
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'patch_imgs': patch_imgs, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}

if __name__=='__main__':

    import torch
    import argparse
    import numpy as np
    import os
    from utils_anyres import get_transform

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='data/visa')
    parser.add_argument('--save_path', type=str, default='save')
    parser.add_argument('--dataset', type=str, default='visa')
    parser.add_argument('--image_size', type=int, default=336)
    parser.add_argument('--patch_image_size', type=int, default=336)
    parser.add_argument('--target_image_size', type=int, default=518)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    preprocess, target_transform = get_transform(args)

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset, args=args)
    batch_data = train_data[0]
    img, patch_img, img_mask = batch_data['img'], batch_data['patch_imgs'], batch_data['img_mask']
    print(img.size(), img_mask.size())