import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import glob
import argparse
import os
import sys
from src import model
from src import util
from src.body import Body
from src.hand import Hand

np.random.seed(0)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.ucf101 import UCF101

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

    return parser



def extract_pose_estimation(dataset, output_dir, sample=None):
    dataset_len = len(dataset)
    for i in range(dataset_len):
        print('current group:', i)
        # if i>4:
        #     break
        vid_dir = dataset[i]['video']
        frames = glob.glob(os.path.join(vid_dir, '*.jpg'))
        append_dir = '/'.join(vid_dir.split('/')[-2:])
        group_output_dir = os.path.join(output_dir, append_dir)

        if sample is not None:
            sampled_idx = np.random.randint(len(frames), size=sample)
            sampled_frames = np.array(frames)[np.array(sampled_idx)]
            frames = sampled_frames

        if not os.path.exists(group_output_dir):
            os.makedirs(group_output_dir)

        for frame in frames:
            print(frame)
            plot_pose_estimation(frame, group_output_dir)

        if i % (dataset_len // 20) == 0:
            print('dataset processed [{}/{}]'.format(i, dataset_len))

# def create_black_canvas(shape):
#     blank_image = np.zeros(shape=shape, dtype=np.uint8)
    # cv2.imshow("Black Blank", blank_image)
    # cv2.imwrite('blank_image.jpg', blank_image)

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
        Dataset = UCF101(vid_path, annotation_path, split, sample_duration, is_master_proc, video_path_formatter)

    else:
        print('not implemented')
        Dataset=None

    return Dataset.get_dataset()

# def plot_pose_estimation_summary():
#


def main():
    args = arg_parser().parse_args()

    output_dir = os.path.join(DATASET_PATH[args.dataset]['OUTPUT_PATH'], args.split) if not args.output else args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vid_path = DATASET_PATH[args.dataset]['VID_PATH']
    annotation_path = DATASET_PATH[args.dataset]['ANNOTATION_PATH']
    dataset = get_vid_dataset(args.dataset, args.split, vid_path, annotation_path)

    test_image = glob.glob(os.path.join(dataset[0]['video'], '*.jpg'))
    if args.image:
        plot_pose_estimation(args.image, output_dir)
    else:
        extract_pose_estimation(dataset, output_dir, sample=args.sample)


if __name__ == '__main__':
    main()
