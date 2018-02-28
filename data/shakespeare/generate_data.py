import argparse
import json
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("raw_metadata")
    parser.add_argument("out_filename", default="data.out")
    args = parser.parse_args()

    metadata = None
    with open(args.raw_metadata, 'r') as f:
        metadata = json.load(f)

    output = ["(sentence, author_id, work_id)"]
    for w in metadata["works"]:
        auth = w["author"]
        with open(w["filename"], 'r') as f:
            output += [(s.strip(), w["author"], w["id"]) for s in f.readlines()]
    random.shuffle(output)
    
    with open(args.out_filename, 'w+') as out:
        for o in output:
            out.write(str(o) + "\n")
