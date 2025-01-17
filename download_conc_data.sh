#!/bin/bash
# All credit goes to copilot 
# RJB 17.01.2024

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <date>"
  exit 1
fi

# Extract the arguments
DATE=$1
OUTFILE=$2

# Construct the URL using the date
URL="https://data.meereisportal.de/data/iup/hdf/s/${DATE:0:4}/asi-AMSR2-s6250-$DATE-v5.4.hdf"

# Use wget to download the file
wget $URL -O "$OUTFILE"

# Check if the download was successful
if [ $? -eq 0 ]; then
  echo "Download successful: $URL"
  echo "Data location: $OUTFILE"
else
  echo "Download failed"
  exit 1
fi
