#!/bin/bash

for i in $(find data/VoxCeleb/prepared_data/train -type f); do
    ext="${i##*.}"
    if [[ $ext = m4a ]]
    then
        p=${i%".m4a"}
        echo $i 
        ffmpeg -v 0  -i $i $p'.wav' </dev/null > /dev/null 2>&1 &
    fi
done

find data/VoxCeleb/prepared_data/train -name "*.m4a" -type f -delete
