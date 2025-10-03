
FOLDER=(
    "flclab-abberior-sted/jmbellavance/2021-06-28 (8)(GlyoxalvsPFAvsmethanol)(goodbad)"
    "flclab-abberior-sted/jmbellavance/2021-06-14 (6)(secondary ab test)(goodbad561)"
    "flclab-abberior-sted/jmbellavance/2021-05-19 (shfus + test glyoxal)(goodbad)(4)"
    "flclab-abberior-sted/jmbellavance/2021-05-07 (good bad STED)"
    "flclab-abberior-sted/jmbellavance/2021-04-29 (GoodvsBadSTED)"
)

for folder in "${FOLDER[@]}"; do
    basename=$(basename "$folder")
    mkdir -p "/home-local2/projects/SSL/denoising-data/lqhq/raw/$basename"
    rclone copy --progress --filter-from "filter-files.txt" "valeria-s3:/$folder" "/home-local2/projects/SSL/denoising-data/lqhq/raw/$basename"
done