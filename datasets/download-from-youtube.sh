#!/bin/sh

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <youtube url> <chunk size in seconds> <dataset path>"
    exit
fi

url=$1
chunk_size=$2
dataset_path=$3

downloaded=".temp"
rm -f $downloaded
format=$(youtube-dl -F $url | grep audio | sed -r 's|([0-9]+).*|\1|g' | tail -n 1)
youtube-dl $url -f $format -o $downloaded

converted=".temp2.wav"
rm -f $converted
ffmpeg -i $downloaded -ac 1 -ab 16k -ar 16000 $converted
rm -f $downloaded

mkdir $dataset_path
length=$(ffprobe -i $converted -show_entries format=duration -v quiet -of csv="p=0")
end=$(echo "$length / $chunk_size - 1" | bc)
echo "splitting..."
for i in $(seq 0 $end); do
    ffmpeg -hide_banner -loglevel error -ss $(($i * $chunk_size)) -t $chunk_size -i $converted "$dataset_path/$i.wav"
done
echo "done"
rm -f $converted
