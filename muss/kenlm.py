# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from pathlib import Path

import kenlm
from tokenizers import SentencePieceBPETokenizer
import sentencepiece as spm  # TODO: JOHAN ?

from mussNL-Fined.muss.utils.helpers import get_temp_filepaths, read_lines, write_lines, log_action, run_command


def train_kenlm_language_model(input_data_paths, output_model_dir):
    output_model_dir = Path(output_model_dir)
    output_model_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_model_dir / 'kenlm_model.arpa'
    with log_action('Training tokenizer'):
        tokenizer = SentencePieceBPETokenizer()
        tokenizer.train([str(path)
                        for path in input_data_paths], vocab_size=20000)
        tokenizer.save(str(output_model_dir), 'spm_tokenizer')
    with log_action('Tokenizing'):
        tokenized_data_paths = get_temp_filepaths(len(input_data_paths))
        for tokenized_data_path, input_data_path in zip(tokenized_data_paths, input_data_paths):
            encodings = tokenizer.encode_batch(read_lines(input_data_path))
            write_lines([' '.join(encoding.tokens)
                        for encoding in encodings], tokenized_data_path)
    with log_action('Training language model'):
        kenlm_path = input(
            'Please provide the path to the lmplz script (install at https://github.com/kpu/kenlm): ')
        command = (
            f'cat {" ".join([str(path) for path in tokenized_data_paths])} | {kenlm_path} -o 3 > {output_model_path}'
        )
        run_command(command, mute=False)
    [path.unlink() for path in tokenized_data_paths]
    return output_model_dir


@lru_cache(maxsize=10)
def get_spm_tokenizer(model_dir):
    assert model_dir.exists(), 'You can download models at https://dl.fbaipublicfiles.com/muss/muss_mining_filtering_kenlm_language_models.tar.gz'
    #merges_file = model_dir / 'spm_tokenizer-merges.txt'
    #vocab_file = model_dir / 'spm_tokenizer-vocab.json'
    # return SentencePieceBPETokenizer(vocab_file=str(vocab_file), merges_file=str(merges_file))
    sp_file = model_dir / 'nl.sp.model'  # TODO: JOHAN ?
    return spm.SentencePieceProcessor(model_file=str(sp_file))


@lru_cache(maxsize=10)
def get_kenlm_model(model_dir):
    assert model_dir.exists(), 'You can download models at https://dl.fbaipublicfiles.com/muss/muss_mining_filtering_kenlm_language_models.tar.gz'
    #model_file = model_dir / 'kenlm_model.arpa'
    # return kenlm.Model(str(model_file))
    model_file = model_dir / 'nl.arpa.bin'
    return kenlm.Model(str(model_file))


def get_kenlm_log_prob(text, model_dir, *args, **kwargs):
    tokenizer = get_spm_tokenizer(model_dir)
    kenlm_model = get_kenlm_model(model_dir)
    encoded_text = ' '.join(tokenizer.encode(text, out_type=str))
    return kenlm_model.score(encoded_text)
