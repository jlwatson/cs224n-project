import html2text
import argparse
import time
from requests_html import HTMLSession
from slugify import slugify

parser = argparse.ArgumentParser()
parser.add_argument('url', help='The Profile url.')
args = parser.parse_args()

h = html2text.HTML2Text()
h.ignore_links = True
h.ignore_images = True

session = HTMLSession()

cursor = 0
while True:
    url = args.url + ';sa=showPosts;start=' + str(cursor)
    print("Crawling", url)
    r = session.get(url)
    td = r.html.find('#bodyarea td', first=True)
    posts = td.find('table[cellpadding="0"]')

    for post in posts:
        url = post.find('.middletext a')[-1].attrs['href']
        content = post.find('.post', first=True).html
        with open(slugify(url) + ".txt", "wb") as f:
            f.write(h.handle(content).encode('utf8'))

    cursor += len(posts)
    
    time.sleep(5)
    if len(posts) < 20:
        break
