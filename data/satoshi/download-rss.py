import html2text
import argparse

from feedparser import parse
from slugify import slugify

parser = argparse.ArgumentParser()
parser.add_argument('url', help='The RSS url.')
args = parser.parse_args()

h = html2text.HTML2Text()
h.ignore_links = True
h.ignore_images = True

url = args.url
while True:
    print ("Parsing", url, "...")
    feed = parse(url)
    for item in feed["items"]:
        with open(slugify(item["link"]) + ".txt", "wb") as f:
            f.write(h.handle(item["summary"]).encode('utf8'))

    if 'feed' in feed and 'links' in feed['feed']:
        next_link = [link['href'] for link in feed['feed']['links'] if link['rel'] == 'next']
        if len(next_link) > 0:
            url = next_link[0]
        else:
            break
    else:
        break
