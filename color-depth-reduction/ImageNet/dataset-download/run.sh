# Replace CSV_FILE with path to dev_dataset.csv or final_dataset.csv
CSV_FILE=final_dataset.csv
# Replace OUTPUT_DIR with path to directory where all images should be stored
OUTPUT_DIR=images_final
# Download images
python3 download_images.py --input_file=${CSV_FILE} --output_dir=${OUTPUT_DIR}
