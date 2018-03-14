from mercury_parser.client import MercuryParser
from slugify import slugify

import os
import argparse
import time

import html2text

from dotenv import load_dotenv, find_dotenv

if __name__ == "__main__":
    load_dotenv(find_dotenv())

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True

    parser = argparse.ArgumentParser()
    parser.add_argument('urls', help='The urls to parse.', metavar='N', nargs='+')
    args = parser.parse_args()

    mercury = MercuryParser(api_key=os.environ['MERCURY_PARSER_KEY'])

    for url in args.urls:
        print("Parsing", url, "...")
        content = h.handle(mercury.parse_article(url).json()['content'])
        with open(slugify(url) + ".txt", "wb") as f:
            f.write(content.encode('utf8'))
        time.sleep(1)
