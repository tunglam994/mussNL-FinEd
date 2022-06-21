# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import gzip
import json
import threading
from tqdm import tqdm

# %%

filepath = 'NL_textdata/nl.txt'

# %%


def write_lines_to_compressed_file(lines, compressed_filepath):
    with gzip.open(compressed_filepath, 'wt', compresslevel=1, encoding="utf-8") as f:
        for line in lines:
            if not line.endswith('\n'):
                line = line + '\n'
            f.write(line)


def json_creator(sentence):
    json_list.append(json.dumps({"raw_content": sentence}, ensure_ascii=False))


# %%
language = 'nl'
for j, chunk in enumerate(tqdm(pd.read_csv(filepath, sep='\t', chunksize=2000000, header=None, error_bad_lines=False))):

    json_list = []
    threads = []
    for i in chunk.values:
        thread = threading.Thread(target=json_creator, args=(i))
        threads.append(thread)
        thread.start()

    for i in threads:
        i.join()

    output_dirs = f'{language}_head_{j:04d}.json.gz'
    write_lines_to_compressed_file(json_list, output_dirs)
