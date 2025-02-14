import os
import random
import numpy as np
import torch
import torch.nn.functional as F

import DHRCLIP_lib
from prompt_DHRCLIP import DHRCLIP_PromptLearner
from dataset_anyres import Dataset
from utils_anyres import get_transform, normalize

from tabulate import tabulate
from logger import get_logger
from tqdm import tqdm
import argparse

from visualization import visualizer
from metrics import image_level_metrics, pixel_level_metrics
from scipy.ndimage import gaussian_filter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DHRCLIP_parameters = {"Abnormal_Prompt_length": args.ab_ctx, "Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    model, _ = DHRCLIP_lib.load("ViT-L/14@336px", device=device, design_details = DHRCLIP_parameters)
    model.eval()

    preprocess, target_transform, patch_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, patch_transform=patch_transform, dataset_name = args.dataset, args=args)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,  persistent_workers=True)
    obj_list = test_data.obj_list


    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    prompt_learner = DHRCLIP_PromptLearner(model.to("cpu"), DHRCLIP_parameters)
    
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    
    prompt_learner.to(device)
    model.to(device)
    
    prompt_learner.eval()
    model.eval()

    model.visual.DAPM_replace(DPAM_layer = args.dpam)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)
    alpha = 0.75

    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        patch_imgs = items['patch_imgs']
        patch_imgs = [patch_img.to(device) for patch_img in patch_imgs]
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, features_list, DPAM_layer = 20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            p_p_features_list = [[],[],[],[],[]]
            p_g_features_list = []
            
            p_p_features_list[0].extend(patch_features)
            p_g_features_list.append(image_features)
            
            for idx, patch_img in enumerate(patch_imgs, start=1):
                p_g_features, p_p_features = model.encode_image(patch_img, args.features_list, DPAM_layer = 20)
                p_p_features_list[idx].extend(p_p_features)
                p_g_features_list.append(p_g_features)  # [8, 768]

            text_probs_list = []
            for p_g_feature in p_g_features_list:
                text_probs = p_g_feature.unsqueeze(1) @ text_features.permute(0, 2, 1)
                text_probs_list.append(text_probs)
            
            text_probs = torch.stack(text_probs_list, dim = 0)
            text_global_probs = text_probs[0, :, ...]
            text_local_probs = text_probs[1:, :, ...]
            text_local_probs = text_local_probs.sum(dim=0) / 4.0

            text_probs = alpha * text_global_probs + (1-alpha) * text_local_probs

            text_probs = (text_probs/0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]

            similarity_map_list = []
            for idx, p_p_feature in enumerate(p_p_features_list):

                similarity_map_list.append([])
                for patch_feature in p_p_feature:

                    patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                    similarity, _ = DHRCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    if idx == 0:
                        similarity_map = DHRCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    else:
                        similarity_map = DHRCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.patch_image_size)
                    anomaly_map = (similarity_map[...,1] + 1 - similarity_map[...,0])/2.0
                    similarity_map_list[idx].append(anomaly_map)

            full_sim_map = [torch.nn.functional.interpolate(ano_map.unsqueeze(0), args.target_image_size, mode='bilinear') for ano_map in similarity_map_list[0]]
            full_similarity_map = torch.stack(full_sim_map).squeeze(1)

            # Patch Aggregation
            full_similarity_map_patch = torch.zeros_like(full_similarity_map) # [8,4,2,518,518]
            full_similarity_map_patch[:, :, 0:args.patch_image_size, 0:args.patch_image_size] += torch.stack(similarity_map_list[1])
            full_similarity_map_patch[:, :, 0:args.patch_image_size, -args.patch_image_size:] += torch.stack(similarity_map_list[2])
            full_similarity_map_patch[:, :, -args.patch_image_size:, 0:args.patch_image_size] += torch.stack(similarity_map_list[3])
            full_similarity_map_patch[:, :, -args.patch_image_size:, -args.patch_image_size:] += torch.stack(similarity_map_list[4])
            
            if args.patch_image_size * 2 > args.target_image_size:
                overlap_patch = args.patch_image_size * 2 - args.target_image_size
                full_similarity_map_patch[:, :, args.patch_image_size-overlap_patch:args.patch_image_size, :] /= 2
                full_similarity_map_patch[:, :, :, args.patch_image_size-overlap_patch:args.patch_image_size] /= 2
            
            full_similarity_map = alpha * full_similarity_map + (1-alpha) * full_similarity_map_patch

            anomaly_map = full_similarity_map.sum(dim = 0)

            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            anomaly_map = torch.stack([torch.from_numpy(normalize(gaussian_filter(i, sigma = args.sigma))) for i in anomaly_map.detach().cpu()], dim = 0 )
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
            visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.target_image_size, args.save_path, cls_name)
            

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    pixel_f1_list = []
    for obj in obj_list:
        table = []
        table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            pixel_f1    = pixel_level_metrics(results, obj, "pixel-f1")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(pixel_f1 * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            pixel_f1_list.append(pixel_f1)
        elif args.metrics == 'image-pixel-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        table_ls.append(table)

    if args.metrics == 'image-level':
        # logger
        table_ls.append(['mean', 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    elif args.metrics == 'pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_f1_list) * 100, decimals=1))
                        ])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'pixel_f1'], tablefmt="pipe")
    elif args.metrics == 'image-pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)), 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'], tablefmt="pipe")
    logger.info("\n%s", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DHRCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/mvtec", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/9_12_4_multiscale_proposed/zero_shot', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/9_12_4_multiscale_proposed/epoch_15.pth', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--patch_image_size", type=int, default=224, help="patch size")
    parser.add_argument("--target_image_size", type=int, default=448, help="patch size")
    
    parser.add_argument("--dpam", type=int, default=20, help="vvclip")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--ab_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=8, help="zero shot")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
