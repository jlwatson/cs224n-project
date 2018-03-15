import html2text
import argparse

from feedparser import parse
from slugify import slugify

def download_author_comments(replies_url, author_names):
    url = replies_url
    while True:
        print ("Parsing comments ", url, "...")
        feed = parse(url)
        for item in feed["items"]:
            if item['author_detail']['name'] in author_names:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('url', help='The RSS url.')
    parser.add_argument('--authornames', nargs='+',
                    help='Names that the author uses in the comments of their blog.')
    args = parser.parse_args()

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True

    url = args.url
    while True:
        print ("Parsing", url, "...")
        feed = parse(url)

        for item in feed["items"]:
            # with open(slugify(item["link"]) + ".txt", "wb") as f:
            #     f.write(h.handle(item["summary"]).encode('utf8'))
            if args.authornames and 'links' in item:
                replies_link = [link['href'] for link in item['links'] if link['rel'] == 'replies']
                if len(replies_link) > 0:
                    download_author_comments(replies_link[0], args.authornames)

        if 'feed' in feed and 'links' in feed['feed']:
            next_link = [link['href'] for link in feed['feed']['links'] if link['rel'] == 'next']
            if len(next_link) > 0:
                url = next_link[0]
            else:
                break
        else:
            break
