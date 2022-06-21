import os
# from pathlib import Path
import pandas as pd
import shutil

from transformers import MarianTokenizer
from datasets import load_dataset


def StoreAndSaveDF(split, directory, target_dir):
    """
    Load the dataset text files which are preprocessed with the corresponding pre-pendings
    (<DEPENDENCYTREEDEPTHRATIO_...> <WORDRANKRATIO_...> <REPLACEONLYLEVENSHTEIN_...> <LENGTHRATIO_...>)
    but not yet tokenized, and save them into a .csv format for easy loading into the HuggingFace dataset type.
    """
    with open(f"{directory}{split}.complex", "r", encoding="utf-8") as f1:
        with open(f"{directory}{split}.simple", "r", encoding="utf-8") as f2:
            complex = f1.readlines()
            simple = f2.readlines()

            complex_clean, simple_clean = [], []
            for line1, line2 in zip(complex, simple):
                complex_clean.append(line1.replace("\n", ""))
                simple_clean.append(line2.replace("\n", ""))

    os.makedirs(target_dir, exist_ok=True)
    df = pd.DataFrame(data={"complex": complex_clean, "simple": simple_clean})
    df.to_csv(f"{target_dir}{split}.csv")
    return df


target_dir = "data/data/"
dir = "data/raw/"
for split in ["train", "test", "valid"]:
    StoreAndSaveDF(split, dir, target_dir)

# Download and save the MarianTokenizer for Dutch to English translation
os.makedirs("tokenizer/", exist_ok=True)
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-nl-en")
tokenizer.save_pretrained("tokenizer/")

# The MarianTokenizer uses a different tokenizer for tokenizing the Dutch language to pieces and its corresponding
# integers than for the English language. As, for simplification, we are translating from complex Dutch to simple Dutch,
# we're going to use the same tokenizer for encoding and decoding (the Dutch tokenizer). For now, for simplicity, we'll
# just remove the English tokenizer from the folder and copy the Dutch tokenizer and rename it to 'target.spm' so that
# it will be used for both encoding and decoding.
os.makedirs("tokenizer/temp/", exist_ok=True)
shutil.move("tokenizer/target.spm", "tokenizer/temp/target.spm")
shutil.copy("tokenizer/source.spm", "tokenizer/target.spm")

tokenizer = MarianTokenizer.from_pretrained("tokenizer/")
def tokenize_function(examples):
    model_inputs = tokenizer(examples["complex"], padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["simple"], padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

data_dir = "data/data/"
target_dir = "data/tokenized/"
os.makedirs(target_dir, exist_ok=True)
column_names = ["Unnamed: 0", "complex", "simple"]

datadict = {"train": f"{data_dir}train.csv", "valid": f"{data_dir}valid.csv", "test": f"{data_dir}test.csv"}
dataset = load_dataset('csv', data_files=datadict)

#dataset = load_dataset('csv', data_files="data/data/test.csv")#, column_names=["complex", "simple"])

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=column_names)
tokenized_dataset.save_to_disk(target_dir)








