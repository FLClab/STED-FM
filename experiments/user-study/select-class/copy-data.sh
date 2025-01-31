
directory="./static/class-guidance"
if [ -d "$directory" ]; then
  echo "$directory does exist."
fi

mkdir -p $directory

rclone copy --progress valeria-s3:flclab-foundation-models/classification-study/class-guidance/ $directory

directory="./static/latent-guidance"
if [ -d "$directory" ]; then
  echo "$directory does exist."
fi

mkdir -p $directory

rclone copy --progress valeria-s3:flclab-foundation-models/classification-study/latent-guidance/ $directory
