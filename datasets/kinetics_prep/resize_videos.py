# Kinetics video resizing script based on
# https://github.com/neuroailab/VIE/tree/master/build_data/kinetics.

# Required libraries: opencv-python, ffmpeg

# Instructions:
# python resize_frames.py 
# --csv_path <path_to_the_downloaded_kinetics_split.csv>
# --video_dir <directory_to_host_downloaded_videos> 
# --out_dir <directory_to_host_the_resized_videos> 
# --len_idx <number_of_videos>
#
# Note: The video frames are resized to have the shortest edge be 256px. Modify
# the resolution_str variable to change this.

import numpy as np
import math
import os
import sys
import argparse
import pdb
from tqdm import tqdm
from multiprocessing import Pool
import functools
import cv2

NUM_THREADS = 10


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to extract the jpgs from videos')
    parser.add_argument(
            '--csv_path',
            default='/mnt/fs1/Dataset/kinetics/kinetics_train.csv',
            type=str, action='store',
            help='Path to the csv containing the information')
    parser.add_argument(
            '--video_dir',
            default='/data5/chengxuz/Dataset/kinetics/vd_dwnld',
            type=str, action='store',
            help='Directory to hold the downloaded videos')
    parser.add_argument(
            '--out_dir',
            default='/data5/chengxuz/Dataset/kinetics/jpgs_extracted_test',
            type=str, action='store',
            help='Directory to hold the extracted jpgs, rescaled')
    parser.add_argument(
            '--sta_idx', default=0, type=int, action='store',
            help='Start index for downloading')
    parser.add_argument(
            '--len_idx', default=100, type=int,
            action='store', help='Length of index of downloading')
    parser.add_argument(
            '--check', default=0, type=int, action='store',
            help='Whether checking the existence')
    return parser


def load_csv(csv_path, return_cate_lbls=False):
    fin = open(csv_path, 'r')
    csv_lines = fin.readlines()
    csv_lines = csv_lines[1:]
    all_data = []

    cate_list = []

    curr_indx = 0

    for curr_line in csv_lines:
        if curr_line[-1]=='\n':
            curr_line = curr_line[:-1]
        line_split = curr_line.split(',')
        curr_cate = line_split[0]
        #curr_cate = curr_cate.replace(' ', '')
        #curr_cate = curr_cate.replace('"', '')
        #curr_cate = curr_cate.replace('(', '')
        #curr_cate = curr_cate.replace(')', '')
        #curr_cate = curr_cate.replace("'", '')

        curr_dict = {
                'cate': curr_cate, 
                'id': line_split[1], 
                'sta': int(line_split[2]), 
                'end': int(line_split[3]), 
                'train': line_split[4], 
                #'flag': int(line_split[5]), 
                'indx': curr_indx}

        if not curr_dict['cate'] in cate_list:
            cate_list.append(curr_dict['cate'])

        curr_dict['cate_lbl'] = cate_list.index(curr_dict['cate'])

        all_data.append(curr_dict)
        curr_indx = curr_indx + 1
    if not return_cate_lbls:
        return all_data
    else:
        return all_data, cate_list


def resize_one_video(curr_indx, args, csv_data):
    curr_data = csv_data[curr_indx]
    mp4_name = '%s_%i_%i.mp4' % (curr_data['id'], curr_data['sta'], curr_data['end'] - curr_data['sta'])
    save_file = os.path.join(args.out_dir, curr_data['cate'], mp4_name)
    mp4_path = os.path.join(args.video_dir, curr_data['cate'], mp4_name)

    if os.path.exists(save_file) and args.check==1:
        print ('Output file exists')
        return

    if not os.path.exists(mp4_path):
        print ('Video does not exist', mp4_path)
        return

    save_folder = os.path.join(args.out_dir, curr_data['cate'])
    if not os.path.exists(save_folder):
        err = os.system('mkdir "%s"' % save_folder)
        if (err != 0):
            print('Failed to create save directory')
            return

    vidcap = cv2.VideoCapture(mp4_path)
    vid_height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vid_width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)

    if vid_width < vid_height:
        resolution_str = '256:-2'  # -2 maintains aspect ratio and keeps even
    else:
        resolution_str = '-2:256'

    cmd = 'ffmpeg -y -i "{}" -vf scale={} "{}" > /dev/null 2>&1'.format(
            mp4_path,
            resolution_str,
            save_file)
    err = os.system(cmd)
    if (err != 0):
        print ('ffmpeg failed, command:',cmd)
        os.system('echo {} >> ids_failed_resizing.txt'.format(mp4_path))


def main():
    parser = get_parser()
    args = parser.parse_args()

    csv_data = load_csv(args.csv_path)
    curr_len = min(len(csv_data) - args.sta_idx, args.len_idx)

    _func = functools.partial(resize_one_video, args=args, csv_data=csv_data)
    p = Pool(NUM_THREADS)
    r = list(tqdm(
        p.imap(
            _func,
            range(args.sta_idx, args.sta_idx + curr_len)),
        total=curr_len))


if __name__ == '__main__':
    main()
