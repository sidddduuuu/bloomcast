#!/usr/bin/env bash
# Re-download all raw public data (FWC cell counts + NDBC buoy 42013).
set -e
mkdir -p data
echo "FWC Historic HAB 2015-2023..."
curl -sL "https://geodata.myfwc.com/api/download/v1/items/eecd72261da2412192123f5a96c4150c/csv?layers=12" -o data/hab_2015_2023.csv
echo "NDBC buoy 42013 winds/temp, 2015-2023..."
for Y in 2015 2016 2017 2018 2019 2020 2021 2022 2023; do
  curl -s "https://www.ndbc.noaa.gov/view_text_file.php?filename=42013h${Y}.txt.gz&dir=data/historical/stdmet/" -o data/42013_${Y}.txt
done
echo "done."
