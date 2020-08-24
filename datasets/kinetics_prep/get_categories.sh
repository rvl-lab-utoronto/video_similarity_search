#!/bin/sh

# Example run: ./get_paths_and_labels.sh /path/to/train_split/ /path/to/categories.csv

path_to_video_split=$1
output_file=$2

label=0
for dir in $path_to_video_split*/
do
    if [ -d "$dir" ]; then
        echo "$(basename "$dir")" >> $output_file
    fi
    label=`expr $label + 1`
done
