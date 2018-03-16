import argparse
import os
import requests
import time

from itertools import filterfalse

def process_comment(source):
    lines = source.splitlines()
    lines = filterfalse(lambda x: x.startswith('>'), lines)
    return '\n'.join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('user', help='The Less Wrong user')
    args = parser.parse_args()

    last_id = ""
    while True:
        url = "http://lesswrong.com/user/%s/comments/.json?after=%s" % (args.user, last_id)
        print("Downloading", url, "...")
        data = requests.get(url).json()['data']['children']

        if len(data) == 0:
            break

        for comment in data:
            with open('lesswrong-' + comment['data']['name'] + ".txt", "wb") as f:
                f.write(process_comment(comment['data']['body']).encode('utf8'))
            last_id = comment['data']['name']

        time.sleep(1)
