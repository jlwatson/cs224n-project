import argparse
import json
from keras.preprocessing import text
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("raw_metadata")
    parser.add_argument("out_filename", default="sp.data")
    # if chunk size is 0, just take line by line
    parser.add_argument("--chunk_size", type=int, default=0)
    args = parser.parse_args()

    metadata = None
    with open(args.raw_metadata, 'r') as f:
        metadata = json.load(f)

    output = []
    for w in metadata["works"]:
        print("processing", w["title"] + "....", end="", flush=True)
        with open(w["filename"], 'r') as f:
            if args.chunk_size == 0: # line by line
                output += [(s.strip(), w["author"], w["id"]) for s in f.readlines()]
            else:
                '''
                alllines = "\n".join([s.strip() for s in f.readlines()])
                words_list = text.text_to_word_sequence(alllines)
                while len(alllines) >= args.chunk_size:
                    output.append((" ".join(alllines[:args.chunk_size]), w["author"], w["id"]))
                    alllines = alllines[args.chunk_size:]
                output.append((" ".join(alllines), w["author"], w["id"]))
                '''
                alllines = [s.strip() for s in f.readlines()]
                while len(alllines) >= args.chunk_size:
                    output.append((" ".join(alllines[:args.chunk_size]), w["author"], w["id"]))
                    alllines = alllines[args.chunk_size:]
                output.append((" ".join(alllines), w["author"], w["id"]))

        print("done.", flush=True)

    random.shuffle(output)

    with open(args.out_filename, 'w+') as f:
        f.write("// Metadata\n")
        f.write(json.dumps(metadata) + "\n\n")
        f.write("// (sentence, author_id, work_id)\n")
        for o in output:
            f.write(str(o) + "\n")

