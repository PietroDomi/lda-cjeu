import html2text, os

def convert():
    
    celexs = os.listdir("data_scraping/data_html")

    converted = os.listdir("data/converted")

    text_maker = html2text.HTML2Text()
    text_maker.ignore_images = True
    text_maker.ignore_links = True

    i = 0

    for celex in celexs:
        i += 1
        if int(celex[1:5]) < int(converted[-1][1:5]):
            with open("data/converted/"+celex[:-5]+".txt","w", encoding='utf-8') as file:
                file.write(text_maker.handle(open("data_scraping/data_html/"+celex,encoding='utf-8').read()))
                print(celex + " converted, " + str(len(celexs)-i) + " remaining")

    return len(celexs)

