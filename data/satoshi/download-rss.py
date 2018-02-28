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

feed = parse(args.url)

for item in feed["items"]:
    with open(slugify(item["link"]) + ".txt", "wb") as f:
        f.write(h.handle(item["summary"]).encode('utf8'))
