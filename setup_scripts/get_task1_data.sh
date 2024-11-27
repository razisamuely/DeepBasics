#!/bin/bash

# Define the URL and output path
URL="https://drive.google.com/uc?id=1s06UaGqogkFmrhnuRK7Tm9fBNi6NYug0"
OUTPUT_PATH="neural_net_from_scratch/data/data.zip"
EXTRACT_PATH="neural_net_from_scratch/data"

# Create the data directory if it doesn't exist
mkdir -p $EXTRACT_PATH

# Download the file using gdown
gdown $URL -O $OUTPUT_PATH

# Unzip the file
unzip $OUTPUT_PATH -d $EXTRACT_PATH

# Remove the zip file
rm $OUTPUT_PATH