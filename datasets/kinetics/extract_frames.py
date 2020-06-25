# Modified Kinetics video frame extraction script from
# https://github.com/neuroailab/VIE/tree/master/build_data/kinetics.

# Required libraries: opencv-python, ffmpeg

# Instructions:
# python extract_frames.py 
# --csv_path <path_to_the_downloaded_kinetics_split.csv>
# --video_dir <directory_to_host_downloaded_videos> 
# --jpg_dir <directory_to_host_the_jpgs> 
# --len_idx <number_of_videos>
#
# Note: The video frames are scaled to have the shortest edge be 320px. Modify
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
            '--jpg_dir',
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
    parser.add_argument(
            '--remove_empty', action='store_true',
            help='Whether just remove empty folders')
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


def extract_one_video(curr_indx, args, csv_data):
    curr_data = csv_data[curr_indx]
    mp4_name = '%s_%i_%i.mp4' % (curr_data['id'], curr_data['sta'], curr_data['end'] - curr_data['sta'])
    save_folder = os.path.join(
            args.jpg_dir,
            curr_data['cate'].replace(' ','\ '), mp4_name.rstrip('.mp4'))
    mp4_path = os.path.join(args.video_dir, curr_data['cate'], mp4_name)
    if args.remove_empty:
        if os.path.exists(save_folder) and not os.path.exists(mp4_path):
            os.rmdir(save_folder)
        return

    if os.path.exists(save_folder) and args.check==1:
        print ('Save folder exists')
        return

    if not os.path.exists(mp4_path):
        print ('Video does not exist', mp4_path)
        return
    os.system('mkdir -p %s' % save_folder)

    vidcap = cv2.VideoCapture(mp4_path)
    vid_height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vid_width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)

    if vid_width < vid_height:
        resolution_str = '320:-1'
    else:
        resolution_str = '-1:320'

    tmpl = '%06d.jpg'
    cmd = 'ffmpeg -i {} -vf scale={},fps=25 {} > /dev/null 2>&1'.format(
            mp4_path.replace(' ','\ '),
            resolution_str,
            os.path.join(save_folder, tmpl))
    err = os.system(cmd)
    if (err != 0):
        print ('ffmpeg failed for', mp4_path.replace(' ', '\ '))


def main():
    parser = get_parser()
    args = parser.parse_args()

    csv_data = load_csv(args.csv_path)
    curr_len = min(len(csv_data) - args.sta_idx, args.len_idx)

    _func = functools.partial(extract_one_video, args=args, csv_data=csv_data)
    p = Pool(NUM_THREADS)
    r = list(tqdm(
        p.imap(
            _func,
            range(args.sta_idx, args.sta_idx + curr_len)),
        total=curr_len))


if __name__ == '__main__':
    main()
