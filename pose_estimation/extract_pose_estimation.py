import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import glob
import argparse
import os
import sys
import matplotlib.pyplot as plt
from src import model
from src import util
from src.body import Body
from src.hand import Hand

SOURCE_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SOURCE_CODE_DIR))

from datasets.ucf101 import UCF101
from misc.upload_gdrive import upload_file_to_gdrive

np.random.seed(0)
body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

DATASET_PATH = {
    'ucf101': {
        'VID_PATH': '/media/diskstation/datasets/UCF101/jpg',
        'ANNOTATION_PATH': '/media/diskstation/datasets/UCF101/json/ucf101_01.json',
        'OUTPUT_PATH': '/media/diskstation/datasets/UCF101/pose'
        },
    'kinetics': {}
}

def arg_parser():
    parser = argparse.ArgumentParser("pose estimation")
    # parser.add_argument(
    #     '--vid_path',
    #     default='/home/sherry/code/pose_repo/pytorch-openpose/images/ucf_frames',
    #     type=str,
    #     help='Path to store the root dir'
    # )
    parser.add_argument(
        '-o',
        '--output',
        default=None,
        type=str,
        help='output directory'
    )
    parser.add_argument(
        '-d',
        '--dataset',
        default='ucf101',
        type=str,
        help='dataset name'
    )
    parser.add_argument(
        '--split',
        default='train',
        type=str,
        help='train/val'
    )
    parser.add_argument(
        '--sample',
        default=None,
        type=int,
        help='randomly sample {?} of frames in a vid group'
    )
    parser.add_argument(
        '-img',
        '--image',
        default=None,
        type=str,
        help='if only want to run on a single image'
    )
    parser.add_argument(
        '-c',
        '--continue_run',
        action='store_true',
        help='if only want to run on a single image'
    )

    return parser



def extract_pose_estimation(dataset, output_dir, sample=None, continue_run=False):
    dataset_len = len(dataset.values())
    # for group in dataset:
    #     if sample is not None:
    #         sampled_idx = np.random.randint(len(dataset[group]), size=sample)
    #         sampled_frames = np.array(dataset[group])[np.array(sampled_idx)]
    #         frames = sampled_frames
    #     else:
    #         frames = dataset[group]
    #
    #     vid_dir = frames[0]['video']
    #     frames = glob.glob(os.path.join(vid_dir, '*.jpg'))
    #     append_dir = '/'.join(vid_dir.split('/')[-2:])
    #     group_output_dir = os.path.join(output_dir, append_dir)
    #
    #     if not os.path.exists(group_output_dir):
    #         os.makedirs(group_output_dir)
    #     elif continue_run:
    #         continue
    #
    #     for frame in frames:
    #         print(frame)
    #         plot_pose_estimation(frame, group_output_dir)
    #
    #     # if i % (dataset_len // 20) == 0:
    #     #     print('dataset processed [{}/{}]'.format(i, dataset_len))
    #


    cur_label = None
    for group in dataset:
        # if sample is not None:
        #     clip = np.random.choice(dataset[group][0], 1)
        clip = dataset[group][0]
        vid_dir = clip['video']
        if clip['label'] != cur_label:
            print('current label: ', clip['label'])
            cur_label = clip['label']

        frames = glob.glob(os.path.join(vid_dir, '*.jpg'))
        append_dir = '/'.join(vid_dir.split('/')[-2:])
        group_output_dir = os.path.join(output_dir, append_dir)

        if sample is not None:
            sampled_idx = np.random.randint(len(frames), size=sample)
            sampled_frames = np.array(frames)[np.array(sampled_idx)]
            frames = sampled_frames

        if not os.path.exists(group_output_dir):
            os.makedirs(group_output_dir)
        elif continue_run:
            continue

        for frame in frames:
            print(frame)
            plot_pose_estimation(frame, group_output_dir)

        # if i % (dataset_len // 20) == 0:
        #     print('dataset processed [{}/{}]'.format(i, dataset_len))


def draw_on_canvas(oriImg, canvas, candidate, subset, output_dir, img_name=None, stickwidth=1):
    canvas = util.draw_bodypose(canvas, candidate, subset, stickwidth=1)
    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        all_hand_peaks.append(peaks)

    canvas = util.draw_handpose(canvas, all_hand_peaks)
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, '{}.png'.format(img_name)))
    # plt.show()


def plot_pose_estimation(test_image, output_dir):
    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    blank_canvas = np.zeros(shape=oriImg.shape, dtype=np.uint8)
    image_name = os.path.basename(test_image).replace('.jpg', '')

    draw_on_canvas(oriImg, canvas, candidate, subset, output_dir, img_name='{}_pose'.format(image_name))
    draw_on_canvas(oriImg, blank_canvas, candidate, subset, output_dir, img_name='{}_skeleton'.format(image_name))


def get_vid_dataset(dataset_name, split, vid_path, annotation_path, sample_duration=16, is_master_proc=True):
    if dataset_name == 'ucf101':
        video_path_formatter = (lambda root_path, label, video_id: root_path + '/' +
                                label + '/' + video_id)
        Dataset = UCF101(vid_path, annotation_path, split, sample_duration, is_master_proc, video_path_formatter).get_dataset()
        ucf_dataset = {}
        for data in Dataset:
            group_name = '_'.join(os.path.basename(data['video']).split('_')[:-1])
            if group_name not in ucf_dataset:
                ucf_dataset[group_name] = []
            ucf_dataset[group_name].append(data)

        #skip the dataset
        for group in ucf_dataset:
            ucf_dataset[group] = np.random.choice(ucf_dataset[group], 1)

        #handling sampling here

        return ucf_dataset
    else:
        print('not implemented')
        Dataset=None
        return None

def plot_pose_estimation_summary(output_dir):
    imgs = []
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            sub_dir = os.path.join(root_dir, dir)
            frames = np.array(glob.glob(os.path.join(sub_dir, '*/*_pose.jpg')))
            imgs.append(np.random.choice(frames, 1))

    fig = plt.figure(0)
    for i in range(50):
        ax = fig.add_subplot(10, 5, i)
        img = plt.imread(imgs[i])
        img_title = os.path.dirname(img).split('/')[-2]
        plt.imshow(img)
        ax.set_title(img_title, fontsize=5, pad=0.3)
        plt.axis('off')

    png_file = os.path.join(output_dir, 'pose_estimation_summary1.png')
    fig.tight_layout(pad=3.5)
    plt.savefig(png_file, dpi=300)
    upload_file_to_gdrive(png_file, 'pose_estimation') #TODO: add pose_estimation folder
    print('figure saved to: {}, and uploaded to GoogleDrive'.format(png_file))



def main():
    args = arg_parser().parse_args()

    output_dir = os.path.join(DATASET_PATH[args.dataset]['OUTPUT_PATH'], args.split) if not args.output else args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vid_path = DATASET_PATH[args.dataset]['VID_PATH']
    annotation_path = DATASET_PATH[args.dataset]['ANNOTATION_PATH']

    # test_image = glob.glob(os.path.join(dataset[0]['video'], '*.jpg'))
    if args.image:
        plot_pose_estimation(args.image, output_dir)
    else:
        # cfg = os.path.join(SOURCE_CODE_DIR, '{}.yaml'.format(args.dataset))
        # data_loader, data = data_loader.build_data_loader(args.split, cfg, triplets=False)
        dataset = get_vid_dataset(args.dataset, args.split, vid_path, annotation_path)
        extract_pose_estimation(dataset, output_dir, sample=args.sample, continue_run=args.continue_run)


if __name__ == '__main__':
    main()
