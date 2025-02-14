import os
import random
import numpy as np
import torch
import torch.nn.functional as F

import DHRCLIP_lib
from prompt_DHRCLIP import DHRCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from dataset_anyres import Dataset
from utils_anyres import get_transform

from logger import get_logger
from tqdm import tqdm
import argparse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):

    logger = get_logger(args.save_path)

    preprocess, target_transform, patch_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DHRCLIP_parameters = {"Abnormal_Prompt_length": args.ab_ctx, "Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = DHRCLIP_lib.load("ViT-L/14@336px", device=device, design_details = DHRCLIP_parameters)
    model.eval()
    
    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, patch_transform=patch_transform, dataset_name = args.dataset, args=args)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,  persistent_workers=True)

    prompt_learner = DHRCLIP_PromptLearner(model.to("cpu"), DHRCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = args.dpam)

    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # hyperparameters
    alpha = 0.75
    
    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()

        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader):
            image      = items['img'].to(device)
            patch_imgs = items['patch_imgs']
            patch_imgs = [patch_img.to(device) for patch_img in patch_imgs]
            label      =  items['anomaly']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5]  = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer = 20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                p_p_features_list = [[],[],[],[],[]]
                
                p_p_features_list[0].extend(patch_features)
                
                for idx, patch_img in enumerate(patch_imgs, start=1):
                    _, p_p_features = model.encode_image(patch_img, args.features_list, DPAM_layer = 20)
                    p_p_features_list[idx].extend(p_p_features)

            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
            
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)

            similarity_map_list   = []
            for idx, p_p_feature in enumerate(p_p_features_list):

                similarity_map_list.append([])
                for patch_feature in p_p_feature:

                    patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
                    similarity, _ = DHRCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    
                    if idx == 0:
                        similarity_map = DHRCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    else:
                        similarity_map = DHRCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.patch_image_size).permute(0, 3, 1, 2)
                    
                    similarity_map_list[idx].append(similarity_map)

            full_sim_map = [torch.nn.functional.interpolate(sim_map, args.target_image_size, mode='bilinear') for sim_map in similarity_map_list[0]]
            full_similarity_map = torch.stack(full_sim_map).permute(1, 0, 2, 3, 4)
            
            # Patch Aggregation
            full_similarity_map_patch = torch.zeros_like(full_similarity_map) # [8,4,2,518,518]
            full_similarity_map_patch[:, :, :, 0:args.patch_image_size, 0:args.patch_image_size] += torch.stack(similarity_map_list[1]).permute(1, 0, 2, 3, 4)
            full_similarity_map_patch[:, :, :, 0:args.patch_image_size, -args.patch_image_size:] += torch.stack(similarity_map_list[2]).permute(1, 0, 2, 3, 4)
            full_similarity_map_patch[:, :, :, -args.patch_image_size:, 0:args.patch_image_size] += torch.stack(similarity_map_list[3]).permute(1, 0, 2, 3, 4)
            full_similarity_map_patch[:, :, :, -args.patch_image_size:, -args.patch_image_size:] += torch.stack(similarity_map_list[4]).permute(1, 0, 2, 3, 4)
            
            if args.patch_image_size * 2 > args.target_image_size:
                overlap_patch = args.patch_image_size * 2 - args.target_image_size
                full_similarity_map_patch[:, :, :, args.patch_image_size-overlap_patch:args.patch_image_size, :] /= 2
                full_similarity_map_patch[:, :, :, :, args.patch_image_size-overlap_patch:args.patch_image_size] /= 2
            
            full_similarity_map = alpha * full_similarity_map + (1-alpha) * full_similarity_map_patch
            full_similarity_map = full_similarity_map.permute(1, 0, 2, 3, 4)

            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...]/0.07
            
            image_loss = F.cross_entropy(text_probs, label.long().to(device))
            image_loss_list.append(image_loss.item())

            loss = 0
            for i in range(len(full_similarity_map)):
                loss += loss_focal(full_similarity_map[i], gt)
                loss += loss_dice(full_similarity_map[i][:, 1, :, :], gt)
                loss += loss_dice(full_similarity_map[i][:, 0, :, :], 1-gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DHRCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoints/9_12_4_multiscale_proposed/', help='path to save results')

    parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")

    parser.add_argument("--dpam", type=int, default=20, help="vvclip")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--ab_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--patch_image_size", type=int, default=224, help="patch size")
    parser.add_argument("--target_image_size", type=int, default=448, help="patch size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
