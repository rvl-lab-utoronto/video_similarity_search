#!/bin/sh

# Example run: ./get_paths_and_labels.sh /path/to/train_split/ /path/to/train.csv

path_to_video_split=$1
output_file=$2

label=0
for dir in $path_to_video_split*/
do
    for file in "$dir"*
    do
        if [ -f "$file" ]; then
            #echo "$file,$(basename "$dir")" >> $output_file
            echo "$file,$label" >> $output_file
        fi
    done
    label=`expr $label + 1`
done
