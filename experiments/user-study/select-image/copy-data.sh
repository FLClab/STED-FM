
directory="./static/attention-maps"
if [ -d "$directory" ]; then
  echo "$directory does exist."
fi

mkdir -p $directory

rclone copy --progress valeria-s3:flclab-foundation-models/attention-map-examples/ $directory
