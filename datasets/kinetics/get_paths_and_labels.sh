#!/bin/sh

# Example run: ./get_paths_and_labels.sh /path/to/train_split/ /path/to/train.csv

path_to_video_split=$1
output_file=$2

for dir in $path_to_video_split*/
do
    for file in "$dir"*
    do
        if [ -f "$file" ]; then
            echo "$file,$(basename "$dir")" >> $output_file
        fi
    done
done
