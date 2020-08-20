#!/bin/sh

# Example run: ./get_paths_and_labels.sh /path/to/train_split/ /path/to/train.csv

path_to_video_split=$1
output_file=$2

label=0
split=$(basename "$path_to_video_split")
for dir in $path_to_video_split*/
do
    categ=$(basename "$dir")
    for vid_dir in "$dir"*
    do
        if [ -d "$vid_dir" ]; then
            #echo "$file,$(basename "$dir")" >> $output_file
            num_frames=$(find "$vid_dir" -maxdepth 1 -type f | grep -c /)
            if [ $num_frames -ne 0 ]; then
                vid=$(basename "$vid_dir")
                echo "$split/$categ/$vid,$num_frames,$label" >> $output_file
                #echo "$vid_dir,$label,$num_frames" >> $output_file
            fi
        fi
    done
    label=`expr $label + 1`
done
