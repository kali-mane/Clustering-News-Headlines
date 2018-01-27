# News Scrapping using BeautifulSoup

from bs4 import BeautifulSoup
import requests
import os


def main():
    #url = "https://timesofindia.indiatimes.com/india"
    #url_google_news = "https://news.google.com/news/headlines?hl=en-IN&gl=IN&ned=in"
    url_google_news = "https://news.google.com/news/headlines?ned=in&hl=en-IN&gl=IN"

    data = requests.get(url_google_news)
    soup = BeautifulSoup(data.content, "lxml")
    # instead of html_parser, lxml can also be used

    path = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(path, 'data\scrapped_headlines.txt')

    links = soup.find_all("a")
    with open(filename, 'w', encoding='utf-8') as f:
        for link in links:
            text = link.text
            headline_length = len(text.split())
            if headline_length > 4:
                f.write(text)
                f.write('\n')
    f.close()


if __name__ == '__main__':
    main()
