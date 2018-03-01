import argparse
import json
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("raw_metadata")
    parser.add_argument("out_filename", default="sp.data")
    args = parser.parse_args()

    metadata = None
    with open(args.raw_metadata, 'r') as f:
        metadata = json.load(f)

    output = []
    for w in metadata["works"]:
        auth = w["author"]
        with open(w["filename"], 'r') as f:
            output += [(s.strip(), w["author"], w["id"]) for s in f.readlines()]
    random.shuffle(output)

    with open(args.out_filename, 'w+') as f:
        f.write("(sentence, author_id, work_id)\n")
        for o in output:
            f.write(str(o) + "\n")

