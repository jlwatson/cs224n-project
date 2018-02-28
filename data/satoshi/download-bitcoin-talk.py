import html2text

from requests_html import HTMLSession
from slugify import slugify

h = html2text.HTML2Text()
h.ignore_links = True
h.ignore_images = True

session = HTMLSession()

r = session.get('https://bitcointalk.org/index.php?action=profile;u=224;sa=showPosts;start=0')
td = r.html.find('#bodyarea td', first=True)
posts = td.find('table[cellpadding="0"]')
print(len(posts))

for post in posts:
    url = post.find('.middletext a')[-1].attrs['href']
    content = post.find('.post', first=True).html
    with open(slugify(url) + ".txt", "wb") as f:
        f.write(h.handle(content).encode('utf8'))
