@ECHO OFF
cd data_scraping
scrapy crawl celex
ECHO Scraping finished
cd ..
python html_converter.py
ECHO Conversion done