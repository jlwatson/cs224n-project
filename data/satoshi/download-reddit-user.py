import argparse
import praw
import os

from slugify import slugify
from itertools import filterfalse
from dotenv import load_dotenv, find_dotenv

def process_comment(source):
    lines = source.splitlines()
    lines = filterfalse(lambda x: x.startswith('>'), lines)
    return '\n'.join(lines)

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser()
    parser.add_argument('user', help='The Reddit user')
    args = parser.parse_args()

    r = praw.Reddit(client_id=os.environ['REDDIT_CLIENT_ID'],
        client_secret=os.environ['REDDIT_CLIENT_SECRET'],
        user_agent='Reddit User Scraper (/u/varunramesh)')

    user = r.redditor(args.user)
    count = 0
    for comment in user.comments.new(limit=None):
        url = 'https://reddit.com' + comment.permalink
        with open(slugify(url) + ".txt", "wb") as f:
            f.write(process_comment(comment.body).encode('utf8'))
        count += 1

    print(count, "posts downloaded...")
