import torchvision.transforms as transforms
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from DHRCLIP_lib.transform import image_transform
from DHRCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD



def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def get_transform(args):
    preprocess = image_transform(args.image_size, is_train=False, mean = OPENAI_DATASET_MEAN, std = OPENAI_DATASET_STD)
    preprocess_patch = image_transform(args.patch_image_size, is_train=False, mean = OPENAI_DATASET_MEAN, std = OPENAI_DATASET_STD)
    target_transform = transforms.Compose([
        transforms.Resize((args.target_image_size, args.target_image_size)),
        transforms.CenterCrop(args.target_image_size),
        transforms.ToTensor()
    ])
    preprocess.transforms[0] = transforms.Resize(size=(args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                    max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size))
    preprocess_patch.transforms[0] = transforms.Resize(size=(args.patch_image_size, args.patch_image_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                    max_size=None, antialias=None)
    preprocess_patch.transforms[1] = transforms.CenterCrop(size=(args.patch_image_size, args.patch_image_size))
    return preprocess, target_transform, preprocess_patch


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches

def resize_and_patch_image(image, resize_resolution=(672, 672), patch_size=336):
    """
    이미지를 주어진 해상도로 리사이즈하고, 패치 크기에 따라 이미지를 나눕니다.

    Args:
        image (PIL.Image.Image): 입력 이미지
        resize_resolution (tuple): 리사이즈할 해상도 (width, height)
        patch_size (int): 패치 크기 (정사각형 패치 크기)

    Returns:
        list: PIL.Image.Image 객체 리스트로 구성된 패치들
    """
    # 이미지 리사이즈
    resized_image = image.resize(resize_resolution)
    
    # 패치 분할
    patches = divide_to_patches(resized_image, patch_size)
    
    return resized_image, patches