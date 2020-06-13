# -*- coding: utf-8 -*-
import scrapy

with open("to_get.txt","r") as file:
    celexs = file.readlines()

class CelexSpider(scrapy.Spider):
    name = 'celex'
    allowed_domains = ['eur-lex.europa.eu/legal-content/IT/TXT/HTML/']
    start_urls = ["https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:" + c[:-1] for c in celexs]

    def parse(self, response):
        page = response.url.split(":")[-1]
        filename = 'data_html/%s.html' % page
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
