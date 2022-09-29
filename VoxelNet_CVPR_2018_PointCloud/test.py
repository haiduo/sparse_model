import os
import argparse

import cv2
cv2.setNumThreads(0)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as DataLoader
from loader.kitti import collate_fn

from config import cfg
from utils.utils import box3d_to_label
from model.model import RPN3D
from loader.kitti import KITTI as Dataset

import torch.profiler
import time
# from google.protobuf import text_format
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description = 'training')

parser.add_argument('--tag', type = str, default = 'default', help = 'log tag')
parser.add_argument('--output_path', type = str, default = './preds', help = 'results output dir')
parser.add_argument('--vis', type = bool, default = True, help = 'set to True if dumping visualization')

parser.add_argument('--batch_size', type = int, default = 2, help = 'batch size')
parser.add_argument('--workers', type = int, default = 0)

parser.add_argument('--resumed_model', type = str, default = 'kitti_Car_best.pth.tar', help = 'if specified, load the specified model')


args = parser.parse_args()


def run():
    args_dic = vars(args)
    for key in args_dic:
        cfg[key] = args_dic[key]

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in cfg.GPU_AVAILABLE)
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Build data loader
    val_dataset = Dataset(os.path.join(cfg.DATA_DIR, 'validation'), shuffle = False, aug = False, is_testset = False)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn,
                                num_workers = args.workers, pin_memory = False)

    # Build model
    model = RPN3D(cfg.DETECT_OBJ)

    # Resume model
    if args.resumed_model:
        model_file = os.path.join(save_model_dir, args.resumed_model)
        if os.path.isfile(model_file):
            checkpoint = torch.load(model_file)
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            print(("=> Loaded checkpoint '{}' (epoch {}, global_counter {})".format(
                args.resumed_model, checkpoint['epoch'], checkpoint['global_counter'])))
        else:
            print(("=> No checkpoint found at '{}'".format(args.resumed_model)))
    else:
        raise Exception('No pre-trained model to test!')

    model = nn.DataParallel(model).cuda()
    # model = model.to(device).eval()

    model.train(False)  # Validation mode
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resuls/infer/torchsparse/all',
        worker_name='worker_numwork2'),
        # record_shapes=True,
        # profile_memory=True,
        # with_stack=True
        ) as prof: 
        schedule_time = time.time()
        with torch.no_grad():
            for (i, val_data) in enumerate(val_dataloader):
                # val_data = val_data.to(device)
                # Forward pass for validation and prediction
                probs, deltas, val_loss, val_cls_loss, val_reg_loss, cls_pos_loss_rec, cls_neg_loss_rec = model(val_data)

                if i >= (1 + 1 + 3) * 2:
                    break
                prof.step()

        print('schedule time:', time.time() - schedule_time)
                # front_images, bird_views, heatmaps = None, None, None
                # if args.vis:
                #     tags, ret_box3d_scores, front_images, bird_views, heatmaps = \
                #         model.module.predict(val_data, probs, deltas, summary = False, vis = True)
                # else:
                #     tags, ret_box3d_scores = model.module.predict(val_data, probs, deltas, summary = False, vis = False)

                # # tags: (N)
                # # ret_box3d_scores: (N, N'); (class, x, y, z, h, w, l, rz, score)
                # for tag, score in zip(tags, ret_box3d_scores):
                #     output_path = os.path.join(args.output_path, 'data', tag + '.txt')
                #     with open(output_path, 'w+') as f:
                #         labels = box3d_to_label([score[:, 1:8]], [score[:, 0]], [score[:, -1]], coordinate = 'lidar')[0]
                #         for line in labels:
                #             f.write(line)
                #         print('Write out {} objects to {}'.format(len(labels), tag))

                # # Dump visualizations
                # if args.vis:
                #     for tag, front_image, bird_view, heatmap in zip(tags, front_images, bird_views, heatmaps):
                #         front_img_path = os.path.join(args.output_path, 'vis', tag + '_front.jpg')
                #         bird_view_path = os.path.join(args.output_path, 'vis', tag + '_bv.jpg')
                #         heatmap_path = os.path.join(args.output_path, 'vis', tag + '_heatmap.jpg')
                #         cv2.imwrite(front_img_path, front_image)
                #         cv2.imwrite(bird_view_path, bird_view)
                #         cv2.imwrite(heatmap_path, heatmap)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    dataset_dir = cfg.DATA_DIR
    val_dir = os.path.join(cfg.DATA_DIR, 'validation')
    save_model_dir = os.path.join('./saved', args.tag)

    # Create output folder
    os.makedirs(args.output_path, exist_ok = True)
    os.makedirs(os.path.join(args.output_path, 'data'), exist_ok = True)
    if args.vis:
        os.makedirs(os.path.join(args.output_path, 'vis'), exist_ok =  True)

    run()