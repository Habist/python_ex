from urllib.request import urlopen
from bs4 import BeautifulSoup

url = "https://news.naver.com/"
html = urlopen(url)
source = html.read()
html.close()

bs = BeautifulSoup(source, "html5lib")

title = bs.find(class_="hdline_article_list")
title_list = title.find_all(class_="hdline_article_tit")

for title_html in title_list:
    if title_html.a.get("href") != "":
        print(title_html.get_text().strip())
