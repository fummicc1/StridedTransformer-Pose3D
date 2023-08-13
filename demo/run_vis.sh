#!/bin/bash

folder=demo/video/
file_path=$(realpath $(dirname $0))

list=$(ls $folder)

for path in $list; do
    python ${file_path}/vis.py --video $path
    # rm $folder/$path
done
