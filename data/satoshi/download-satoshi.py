import requests
import html2text

h = html2text.HTML2Text()
h.ignore_links = True
h.ignore_images = True

emails = requests.get('https://raw.githubusercontent.com/NakamotoInstitute/nakamotoinstitute.org/master/data/satoshiemails.json').json()

for i, email in enumerate(emails['emails']):
    with open("email-%d.txt" % i, "wb") as f:
        f.write(email['Text'].encode('utf8'))

posts = requests.get('https://raw.githubusercontent.com/NakamotoInstitute/nakamotoinstitute.org/master/data/satoshiposts.json').json()
for i, post in enumerate(posts['posts']):
    with open("post-%d.txt" % i, "wb") as f:
        f.write(h.handle(post['post']).encode('utf8'))
