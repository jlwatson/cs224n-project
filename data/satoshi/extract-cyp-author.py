import click
import re

from itertools import filterfalse

MAIL_HEADER = re.compile('^From cypherpunks@MHonArc.venona.*$', re.MULTILINE)
MESSAGE_ID = re.compile('^Message-ID: <(.*)>$', re.MULTILINE)
MESSAGE_FROM = re.compile('^From: (.*)$', re.MULTILINE)

PGP_SIG_BEGIN = "-----BEGIN PGP SIGNATURE-----"
PGP_SIG_END = "-----END PGP SIGNATURE-----"

REMOVE_PREFIX = set(s.lower() for s in [
    'From:',
    'To:',
    'Subject:',
    'Bcc:',
    'Message-ID:',
    '>',
    "Hal Finney",
    "Nick Szabo"
])

REMOVE_LINES = set(s.lower() for s in [
    "hfinney@shell.portal.com",
    "Distribution:",
    "  CYPHERPUNKS >INTERNET:CYPHERPUNKS@TOAD.COM",
    "74076.1041@compuserve.com",
    "-----BEGIN PGP SIGNED MESSAGE-----",
    "Hal",
    "szabo@netcom.com",
    "David",
    "Thanks,"
])

def process_email(email):
    lines = email.splitlines()

    if PGP_SIG_BEGIN in lines and PGP_SIG_END in lines:
        lines = lines[:lines.index(PGP_SIG_BEGIN)] + lines[lines.index(PGP_SIG_END) + 1:]

    lines = filterfalse(lambda x: x.lower() in REMOVE_LINES, lines)
    lines = filterfalse(lambda x: any(x.lower().startswith(prefix) for prefix in REMOVE_PREFIX), lines)

    return '\n'.join(lines).strip()

@click.command()
@click.option('--archive', type=click.File('r', encoding='latin-1'), required=True)
@click.option('--author', required=True)
def extract_author(archive, author):
    """Extract author's emails from Cypherpunk archive."""

    author = author.lower()
    content = archive.read()
    mails = MAIL_HEADER.split(content)
    print(len(mails), "mails in archive.")

    for mail in mails:
        mail = mail.strip()
        if mail == "":
            continue

        message_id = MESSAGE_ID.search(mail).group(1)
        message_author = MESSAGE_FROM.search(mail).group(1).strip().lower()

        if message_author == author:
            try:
                header_end = mail.index('\n\n')
            except:
                print("Couldn't find content for message:", message_id)
                continue

            content = process_email(mail[header_end:].strip())
            print("Saving message:", message_id)
            with open(message_id + ".txt", "wb") as f:
                f.write(content.encode('utf8'))
if __name__ == '__main__':
    extract_author()

74076
