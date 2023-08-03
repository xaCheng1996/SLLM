import os
import sys
import time
import argparse

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import torchvision
import torch.optim as optim
from utils.utils import AverageMeter, reduce_tensor, accuracy
from utils.logger import setup_logger
import clip
from tqdm import tqdm
from pathlib import Path
import yaml
import pprint
from dotmap import DotMap
import numpy as np
import wandb
import datetime
import shutil
from contextlib import suppress

from datasets import Video_dataset
from modules.video_clip import video_header, VideoCLIP
from utils.Augmentation import get_augmentation
from utils.solver import _lr_scheduler
from modules.text_prompt import text_prompt
import torch.nn.functional as F


def epoch_saving(epoch, model, optimizer, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)  # just change to your preferred folder/filename


def best_saving(working_dir, epoch, model, optimizer):
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, best_name)  # just change to your preferred folder/filename


def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, default='clip.yaml', help='global config file')
    parser.add_argument('--log_time', default='001')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    args = parser.parse_args()
    return args


def main(args):
    global best_prec1
    """ Training Program """

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    working_dir = os.path.join('./Text2vis_model/', config['data']['dataset'], config['network']['arch'],
                               config['data']['exp_name'])

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('train.py', working_dir)
    wandb.init(dir='./wandb',
               project=config['network']['Project'],
               name='{}_{}_{}_{}_{}'.format(args.log_time,
                                            config['network']['type'],
                                            config['network']['arch'],
                                            config['data']['dataset'],
                                            config['data']['exp_name']))

    # build logger, print env and config
    logger = setup_logger(output=working_dir,
                          name=f'Text4Vis')
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))
    logger.info("------------------------------------")
    pp = pprint.PrettyPrinter(indent=4)
    logger.info(pp.pformat(config))
    logger.info("------------------------------------")
    logger.info("storing name: {}".format(working_dir))

    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True

    # fix the seed for reproducibility
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # get fp16 model and weight
    model, clip_state_dict = clip.load(
        config.network.arch,
        device='cpu', jit=False,
        internal_modeling=config.network.tm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st=config.network.joint_st)  # Must set jit=False for training

    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)

    logger.info('train transforms: {}'.format(transform_train.transforms))
    logger.info('val transforms: {}'.format(transform_val.transforms))

    video_head = video_header(
        config.network.sim_header,
        clip_state_dict)

    if args.precision == "amp" or args.precision == "fp32":
        model = model.float()

    train_data = Video_dataset(
        config.data.train_root, config.data.train_list,
        config.data.label_list, num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
        transform=transform_train, dense_sample=config.data.dense, use_restoration=config.network.use_restoration)

    train_loader = DataLoader(train_data,
                              batch_size=config.data.batch_size, num_workers=config.data.workers,
                              shuffle=True, pin_memory=False, drop_last=True)

    val_data = Video_dataset(
        config.data.val_root, config.data.val_list, config.data.label_list,
        random_shift=False, num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl,
        transform=transform_val, dense_sample=config.data.dense, use_restoration=config.network.use_restoration)

    val_loader = DataLoader(val_data,
                            batch_size=config.data.batch_size, num_workers=config.data.workers,
                            shuffle=False, pin_memory=False, drop_last=True)

    classes, _, text_dict = text_prompt(train_data, prompt_mode=config.network.text_moda)

    class_feats_file = 'text_feats_{}_{}.pt'.format(config['data']['dataset'], config['network']['arch']).replace('/',
                                                                                                                  '')
    if os.path.isfile(class_feats_file):
        logger.info('=> load classes features from {}'.format(class_feats_file))
        classes_features = torch.load(class_feats_file)
    else:
        model.eval()
        with torch.no_grad():
            classes_features = model.encode_text(classes)  # [n_class dim]


    model_full = VideoCLIP(model, video_head, config.data.num_feature, config.network.use_restoration)

    criterion = torch.nn.CrossEntropyLoss()
    criterion_kl = torch.nn.KLDivLoss(reduction="batchmean")
    criterion_mse = torch.nn.MSELoss()

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            logger.info("=> loading checkpoint '{}'".format(config.pretrain))
            checkpoint = torch.load(config.pretrain, map_location='cpu')
            model_full.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume, map_location='cpu')
            model_full.load_state_dict(update_dict(checkpoint['model_state_dict']))
            start_epoch = checkpoint['epoch'] + 1
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(config.evaluate, checkpoint['epoch']))
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.pretrain))

    if config.network.test:
        filename = config.test
        print(("=> loading test checkpoint '{}'".format(filename)))
        checkpoint = torch.load(filename)
        visual_state_dict = {}
        fusion_state_dict = {}
        logit_scale_state_dict = {}
        restore_state_dict = {}
        res_frame_emb = {}
        for k, v in checkpoint['model_state_dict'].items():
            if 'visual' in k:
                visual_state_dict[k[7:]] = v
            elif 'logit_scale' in k:
                logit_scale_state_dict[k] = v
            elif 'fusion' in k:
                fusion_state_dict[k[13:]] = v
            elif 'restore_model' in k:
                restore_state_dict[k[14:]] = v
            else:
                res_frame_emb[k] = v
        model_full.visual.load_state_dict(visual_state_dict)
        model_full.logit_scale.data = logit_scale_state_dict['logit_scale']
        model_full.fusion_model.load_state_dict(fusion_state_dict)
        if config.network.use_restoration:
            model_full.restore_model.load_state_dict(restore_state_dict)
            model_full.restore_frame_position_embeddings.data = res_frame_emb['restore_frame_position_embeddings.weight']
        del checkpoint

    if config.network.shifter and not config.resume:
        filename = config.shifter
        print(("=> loading base checkpoint '{}'".format(filename)))
        checkpoint = torch.load(filename)
        visual_state_dict = {}
        fusion_state_dict = {}
        logit_scale_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if 'visual' in k:
                visual_state_dict[k[7:]] = v
            elif 'logit_scale' in k:
                logit_scale_state_dict[k] = v
            elif 'fusion' in k:
                fusion_state_dict[k[13:]] = v
        model_full.visual.load_state_dict(visual_state_dict)
        model_full.logit_scale.data = logit_scale_state_dict['logit_scale']
        model_full.fusion_model.load_state_dict(fusion_state_dict)
        del checkpoint

    clip_params = []
    restore_params = []
    other_params = []
    for name, param in model_full.named_parameters():
        if 'visual' in name and 'control_point' not in name:
            clip_params.append(param)
        elif 'logit_scale' in name:
            clip_params.append(param)
        elif 'restore' in name:
            restore_params.append(param)
        else:
            other_params.append(param)

    if config.network.shifter:
        optimizer = optim.AdamW([
                                {'params': other_params, 'lr': config.solver.lr * config.solver.clip_ratio},
                                {'params': restore_params, 'lr': config.solver.lr},
                                ],
                                betas=(0.9, 0.999), lr=config.solver.lr, eps=1e-8,
                                weight_decay=config.solver.weight_decay)
    else:
        optimizer = optim.AdamW([{'params': clip_params, 'lr': config.solver.lr * config.solver.clip_ratio},
                                 {'params': other_params, 'lr': config.solver.lr}],
                                betas=(0.9, 0.999), lr=config.solver.lr, eps=1e-8,
                                weight_decay=config.solver.weight_decay)

    lr_scheduler = _lr_scheduler(config, optimizer)

    model_full = model_full.cuda()

    best_prec1 = 0.0
    if config.network.test:
        prec1 = validate(0, val_loader, device, model_full, config, classes_features, logger)
        best_prec1 = 0.0
        logger.info('Testing: {}/{}'.format(prec1, best_prec1))
        exit()

    scaler = GradScaler() if args.precision == "amp" else None



    for epoch in range(start_epoch, config.solver.epochs):

        classes_features = classes_features.cuda()
        if config.network.use_restoration:
            train(model_full, train_loader, optimizer, criterion, scaler,
                  epoch, device, lr_scheduler, config, classes_features, logger, criterion_kl=criterion_kl, criterion_mse=criterion_mse)
        else:
            train(model_full, train_loader, optimizer, criterion, scaler,
                  epoch, device, lr_scheduler, config, classes_features, logger)

        if (epoch + 1) % config.logging.eval_freq == 0:  # and epoch>0
            prec1 = validate(epoch, val_loader, device, model_full, config, classes_features, logger)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            logger.info('Testing: {}/{}'.format(prec1, best_prec1))
            filename = "{}/last_model.pt".format(working_dir)
            logger.info('Saving: {}'.format(filename))

            epoch_saving(epoch, model_full, optimizer, filename)
            if is_best:
                best_saving(working_dir, epoch, model_full, optimizer)


def train(model, train_loader, optimizer, criterion, scaler,
          epoch, device, lr_scheduler, config, text_embedding, logger, criterion_kl=None, criterion_mse=None):
    """ train a epoch """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_kl = AverageMeter()

    model.train()
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    end = time.time()
    t_pre = 0
    for i, (images, list_id, restoration_frame) in enumerate(tqdm(train_loader)):

        optimizer.zero_grad()  # reset gradient
        data_time.update(time.time() - end)
        # b t3 h w
        images = images.view((-1, config.data.num_feature, 3) + images.size()[-2:])  # bt 3 h w
        b, t, c, h, w = images.size()
        images = images.view(-1, c, h, w)
        images = images.cuda()

        if config.network.use_restoration:
            restoration_frame = restoration_frame.view((-1, config.data.num_feature - 1, 3) + images.size()[-2:])
            _, _, c, h, w = restoration_frame.size()
            restoration_frame = restoration_frame.view(-1, c, h, w)
            restoration_frame = restoration_frame.cuda()

        with autocast():
            if config.network.use_restoration:
                logits, image_restoration_embedding, image_restoration_target = \
                    model(images, text_embedding, restoration_frame)

                restoration_kl = F.log_softmax(
                    image_restoration_embedding.view(b * (t - 1) * config.data.num_restoration, -1), dim=1)
                restoration_target_kl = F.softmax(
                    image_restoration_target.view(b * (t - 1) * config.data.num_restoration, -1), dim=1)
                loss_kl = criterion_kl(restoration_kl, restoration_target_kl)

                loss = criterion(logits, list_id.to(device)) + loss_kl
                losses_kl.update(loss_kl.item(), logits.size(0))

            else:
                logits = model(images, text_embedding)  # B 400
                loss = criterion(logits, list_id.to(device))

            # loss regularization
            loss = loss / config.solver.grad_accumulation_steps

            if scaler is not None:
                # back propagation
                scaler.scale(loss).backward()

                if (i + 1) % config.solver.grad_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    # optimizer.zero_grad()  # reset gradient

            else:
                # back propagation
                loss.backward()
                if (i + 1) % config.solver.grad_accumulation_steps == 0:
                    optimizer.step()  # update param

            if config.solver.type != 'monitor':
                if (i + 1) == 1 or (i + 1) % 10 == 0:
                    lr_scheduler.step(epoch + i / len(train_loader))

            losses.update(loss.item(), logits.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            cur_iter = epoch * len(train_loader) + i
            max_iter = config.solver.epochs * len(train_loader)
            eta_sec = batch_time.avg * (max_iter - cur_iter + 1)
            eta_sec = str(datetime.timedelta(seconds=int(eta_sec)))

            if i % config.logging.print_freq == 0:
                wandb.log({"lr": optimizer.param_groups[-1]['lr']})
                wandb.log({"loss_avg": losses.avg})
                t_pos = time.time()
                wandb.log({'vs': (b * config.logging.print_freq) / (t_pos - t_pre)})
                t_pre = time.time()


def validate(epoch, val_loader, device, model, config, text_embedding, logger):
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    t_pos = 0
    t_pre = 0
    with torch.no_grad():
        simi = []
        for i, (image, class_id, rest_img) in enumerate(tqdm(val_loader)):
            image = image.view((-1, config.data.num_feature, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            text_embedding = text_embedding.to(device)
            image = image.to(device).view(-1, c, h, w)

            if config.network.use_restoration:

                restoration_frame = rest_img.view((-1, config.data.num_feature - 1, 3) + image.size()[-2:])
                _, _, c, h, w = restoration_frame.size()
                restoration_frame = restoration_frame.view(-1, c, h, w)
                restoration_frame = restoration_frame.cuda()

                image_embedding = model.test(image, restoration_frame)
            else:
                image_embedding = model.test(image)


            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_embedding @ text_embedding.T)

            prec = accuracy(similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))
            if i % 10 == 0:
                t_pos = time.time()
                wandb.log({'val vs': (b * 10) / (t_pos - t_pre)})
                t_pre = time.time()
    wandb.log({"epoch": epoch})
    wandb.log({"top1": top1.avg})
    wandb.log({"top5": top5.avg})
    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                 .format(top1=top1, top5=top5)))
    return top1.avg


if __name__ == '__main__':
    args = get_parser()
    main(args)
