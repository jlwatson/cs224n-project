import requests
import html2text

from itertools import filterfalse
from pyquery import PyQuery as pq

def process_email(email):
    lines = email.splitlines()

    lines = filterfalse(lambda x: x == 'Satoshi Nakamoto' or x == 'Satoshi', lines)
    lines = filterfalse(lambda x: x == '---------------------------------------------------------------------', lines)
    lines = filterfalse(lambda x: x == 'The Cryptography Mailing List', lines)
    lines = filterfalse(lambda x: x.startswith('Unsubscribe by sending "unsubscribe cryptography" to'), lines)

    lines = filterfalse(lambda x: x.startswith('>'), lines)
    lines = filterfalse(lambda x: x.endswith('wrote:'), lines)
    return '\n'.join(lines)

def process_post(post_html):
    post = pq(post_html)
    post.remove('div.quoteheader')
    print(post.html(method='html'))
    return post.html(method='html')

if __name__ == "__main__":
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True

    emails = requests.get('https://raw.githubusercontent.com/NakamotoInstitute/nakamotoinstitute.org/master/data/satoshiemails.json').json()

    for i, email in enumerate(emails['emails']):
        with open("email-%d.txt" % i, "wb") as f:
            f.write(process_email(email['Text']).encode('utf8'))

    posts = requests.get('https://raw.githubusercontent.com/NakamotoInstitute/nakamotoinstitute.org/master/data/satoshiposts.json').json()
    for i, post in enumerate(posts['posts']):
        with open("post-%d.txt" % i, "wb") as f:
            f.write(h.handle(process_post(post['post'])).encode('utf8'))
