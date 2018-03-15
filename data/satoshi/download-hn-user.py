import argparse
import requests
import time

import html2text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('user', help='The Hackernews username')
    args = parser.parse_args()

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True

    user = requests.get('https://hacker-news.firebaseio.com/v0/user/%s.json' % args.user).json()
    for item in user['submitted']:
        print("Requesting HN item", item, "...")
        data = requests.get('https://hacker-news.firebaseio.com/v0/item/%s.json' % item).json()
        if data['type'] == 'comment':
            with open("hn-comment-%d.txt" % item, "wb") as f:
                f.write(h.handle(data['text']).encode('utf8'))

        time.sleep(0.5)
