# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 20:15:01 2022

@author: johan
"""

import shutil
import random
import shlex
from pathlib import Path
import re
import tempfile
from urllib.request import urlretrieve
import os

import bz2
import gzip
import tarfile
import zipfile

import tqdm
import ast

import sys
from contextlib import contextmanager
from types import MethodType
import time
from functools import wraps

from fairseq_cli import train

# =============================================================================
# from muss.utils.helpers import (
#     log_std_streams,
#     # lock_directory,
#     # create_directory_or_skip,
#     # yield_lines,
#     # write_lines,
#     mock_cli_args,
#     # create_temp_dir,
#     # mute,
#     args_dict_to_str,
#     print_running_time,
# )
# =============================================================================

from muss.text import remove_multiple_whitespaces
from muss.utils.training import clear_cuda_cache

#from muss.mining.training import get_mbart_kwargs
# %%


@contextmanager
def redirect_streams(source_streams, target_streams):
    # We assign these functions before hand in case a target stream is also a source stream.
    # If it's the case then the write function would be patched leading to infinie recursion
    target_writes = [target_stream.write for target_stream in target_streams]
    target_flushes = [target_stream.flush for target_stream in target_streams]

    def patched_write(self, message):
        for target_write in target_writes:
            target_write(message)

    def patched_flush(self):
        for target_flush in target_flushes:
            target_flush()

    original_source_stream_writes = [
        source_stream.write for source_stream in source_streams]
    original_source_stream_flushes = [
        source_stream.flush for source_stream in source_streams]
    try:
        for source_stream in source_streams:
            source_stream.write = MethodType(patched_write, source_stream)
            source_stream.flush = MethodType(patched_flush, source_stream)
        yield
    finally:
        for source_stream, original_source_stream_write, original_source_stream_flush in zip(
            source_streams, original_source_stream_writes, original_source_stream_flushes
        ):
            source_stream.write = original_source_stream_write
            source_stream.flush = original_source_stream_flush


@contextmanager
def log_std_streams(filepath):
    log_file = open(filepath, 'w', encoding='utf-8')
    try:
        with redirect_streams(source_streams=[sys.stdout], target_streams=[log_file, sys.stdout]):
            with redirect_streams(source_streams=[sys.stderr], target_streams=[log_file, sys.stderr]):
                yield
    finally:
        log_file.close()


def arg_name_python_to_cli(arg_name, cli_sep='-'):
    arg_name = arg_name.replace('_', cli_sep)
    return f'--{arg_name}'


def kwargs_to_cli_args_list(kwargs, cli_sep='-'):
    cli_args_list = []
    for key, val in kwargs.items():
        key = arg_name_python_to_cli(key, cli_sep)
        if isinstance(val, bool):
            cli_args_list.append(str(key))
        else:
            if isinstance(val, str):
                # Add quotes around val
                assert "'" not in val
                val = f"'{val}'"
            cli_args_list.extend([str(key), str(val)])
    return cli_args_list


def args_dict_to_str(args_dict, cli_sep='-'):
    return ' '.join(kwargs_to_cli_args_list(args_dict, cli_sep=cli_sep))

# %%


@contextmanager
def log_action(action_description):
    start_time = time.time()
    print(f'{action_description}...')
    try:
        yield
    except BaseException as e:
        print(f'{action_description} failed after {time.time() - start_time:.2f}s.')
        raise e
    print(f'{action_description} completed after {time.time() - start_time:.2f}s.')


def print_running_time(func):
    '''Decorator to print running time of function for logging purposes'''

    @wraps(func)  # To preserve the name and path for pickling purposes
    def wrapped_func(*args, **kwargs):
        function_name = getattr(func, '__name__', repr(func))
        with log_action(function_name):
            return func(*args, **kwargs)

    return wrapped_func
# %%


@contextmanager
def mock_cli_args(args):
    current_args = sys.argv
    sys.argv = sys.argv[:1] + args
    yield
    sys.argv = current_args


# %%
TEMP_DIR = None


def get_temp_filepath(create=False):
    global TEMP_DIR
    temp_filepath = Path(tempfile.mkstemp()[1])
    if TEMP_DIR is not None:
        temp_filepath.unlink()
        temp_filepath = TEMP_DIR / temp_filepath.name
        temp_filepath.touch(exist_ok=False)
    if not create:
        temp_filepath.unlink()
    return temp_filepath


def reporthook(count, block_size, total_size):
    # Download progress bar
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size_mb = count * block_size / (1024 * 1024)
    speed = progress_size_mb / duration
    percent = int(count * block_size * 100 / total_size)
    msg = f'\r... {percent}% - {int(progress_size_mb)} MB - {speed:.2f} MB/s - {int(duration)}s'
    sys.stdout.write(msg)


def download(url, destination_path=None, overwrite=True):
    if destination_path is None:
        destination_path = get_temp_filepath()
    if not overwrite and destination_path.exists():
        return destination_path
    print('Downloading...')
    try:
        urlretrieve(url, destination_path, reporthook)
        sys.stdout.write('\n')
    except (Exception, KeyboardInterrupt, SystemExit):
        print('Rolling back: remove partially downloaded file')
        os.remove(destination_path)
        raise
    return destination_path


def untar(compressed_path, output_dir):
    with tarfile.open(compressed_path) as f:
        f.extractall(output_dir)


def unzip(compressed_path, output_dir):
    with zipfile.ZipFile(compressed_path, 'r') as f:
        f.extractall(output_dir)


def ungzip(compressed_path, output_dir):
    filename = os.path.basename(compressed_path)
    assert filename.endswith('.gz')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename[:-3])
    with gzip.open(compressed_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def unbz2(compressed_path, output_dir):
    extract_filename = os.path.basename(compressed_path).replace('.bz2', '')
    extract_path = os.path.join(output_dir, extract_filename)
    with bz2.BZ2File(compressed_path, 'rb') as compressed_file, open(extract_path, 'wb') as extract_file:
        for data in tqdm(iter(lambda: compressed_file.read(1024 * 1024), b'')):
            extract_file.write(data)


def move_with_overwrite(source_path, target_path):
    if os.path.isfile(target_path):
        os.remove(target_path)
    if os.path.isdir(target_path) and os.path.isdir(source_path):
        shutil.rmtree(target_path)
    shutil.move(source_path, target_path)


def extract(filepath, output_dir):
    output_dir = Path(output_dir)
    # Infer extract function based on extension
    extensions_to_functions = {
        '.tar.gz': untar,
        '.tar.bz2': untar,
        '.tgz': untar,
        '.zip': unzip,
        '.gz': ungzip,
        '.bz2': unbz2,
    }

    def get_extension(filename, extensions):
        possible_extensions = [
            ext for ext in extensions if filename.endswith(ext)]
        if len(possible_extensions) == 0:
            raise Exception(f'File {filename} has an unknown extension')
        # Take the longest (.tar.gz should take precedence over .gz)
        return max(possible_extensions, key=lambda ext: len(ext))

    filename = os.path.basename(filepath)
    extension = get_extension(filename, list(extensions_to_functions))
    extract_function = extensions_to_functions[extension]

    # Extract files in a temporary dir then move the extracted item back to
    # the ouput dir in order to get the details of what was extracted
    tmp_extract_dir = Path(tempfile.mkdtemp())
    # Extract
    extract_function(filepath, output_dir=tmp_extract_dir)
    extracted_items = os.listdir(tmp_extract_dir)
    output_paths = []
    for name in extracted_items:
        extracted_path = tmp_extract_dir / name
        output_path = output_dir / name
        move_with_overwrite(extracted_path, output_path)
        output_paths.append(output_path)
    return output_paths


def download_and_extract(url):
    tmp_dir = Path(tempfile.mkdtemp())
    compressed_filename = url.split('/')[-1]
    compressed_filepath = tmp_dir / compressed_filename
    download(url, compressed_filepath)
    print('Extracting...')
    extracted_paths = extract(compressed_filepath, tmp_dir)
    compressed_filepath.unlink()
    return extracted_paths


MODELS_DIR = Path('./resources/models/')


def prepare_mbart_model():
    mbart_dir = MODELS_DIR / 'mbart'
    if not mbart_dir.exists():
        url = 'https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.CC25.tar.gz'
        shutil.move(download_and_extract(url)[0], mbart_dir)
    return mbart_dir

# %%


def add_dicts(*dicts):
    return {k: v for dic in dicts for k, v in dic.items()}


def arg_name_cli_to_python(arg_name, cli_sep='-'):
    assert arg_name.startswith('--')
    arg_name = arg_name.strip('-').replace(cli_sep, '_')
    return arg_name


def failsafe_ast_literal_eval(expression):
    try:
        return ast.literal_eval(expression.replace('PosixPath', ''))
    except (SyntaxError, ValueError):
        return expression


def cli_args_list_to_kwargs(cli_args_list):
    kwargs = {}
    i = 0
    while i < len(cli_args_list) - 1:
        assert cli_args_list[i].startswith('--'), cli_args_list[i]
        key = arg_name_cli_to_python(cli_args_list[i])
        next_element = cli_args_list[i + 1]
        if next_element.startswith('--'):
            kwargs[key] = True
            i += 1
        else:
            try:
                kwargs[key] = failsafe_ast_literal_eval(next_element)
            except (SyntaxError, ValueError):
                kwargs[key] = cli_args_list[i + 1]
            i += 2
    return kwargs


def args_str_to_dict(args_str):
    return cli_args_list_to_kwargs(shlex.split(args_str))

# %%


def get_mbart_kwargs(dataset, language, use_access, use_short_name=False):
    mbart_dir = prepare_mbart_model()
    mbart_path = mbart_dir / 'model.pt'
    # source_lang = f'{language}_XX'
    # target_lang = f'{language}_XX'
    source_lang = 'complex'
    target_lang = 'simple'
    kwargs = {
        'dataset': dataset,
        'metrics_coefs': [0, 1, 0],
        'parametrization_budget': 128,
        # 'predict_files': get_predict_files(language),
        'preprocessors_kwargs': {
            'SentencePiecePreprocessor': {
                'sentencepiece_model_path': mbart_dir / 'sentence.bpe.model',
                'tokenize_special_tokens': True,
                # 'vocab_size': 32000,
                # 'input_filepaths': [
                #    get_data_filepath(dataset, 'train', 'complex'),
                #    get_data_filepath(dataset, 'train', 'simple'),
                # ],
            },
        },
        'preprocess_kwargs': {
            'dict_path': mbart_dir / 'dict.txt',
            'source_lang': source_lang,
            'target_lang': target_lang,
        },
        'train_kwargs': add_dicts(
            {'ngpus': 8},
            # args_str_to_dict(
                # f'''--restore-file {mbart_path}  --arch mbart_large --task translation_from_pretrained_bart  --source-lang {source_lang} --target-lang {target_lang}  --encoder-normalize-before --decoder-normalize-before --criterion label_smoothed_cross_entropy --label-smoothing 0.2  --dataset-impl mmap --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --total-num-update 40000 --dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 --max-tokens 1024 --update-freq 2 --log-format simple --log-interval 2 --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
     # --layernorm-embedding  --ddp-backend no_c10d'''
            # ),
             args_str_to_dict(
                 f'''--restore-file {mbart_path}  --arch mbart_large --task translation_from_pretrained_bart  --source-lang {source_lang} --target-lang {target_lang}  --encoder-normalize-before --decoder-normalize-before --criterion label_smoothed_cross_entropy --label-smoothing 0.2  --dataset-impl mmap --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 2500 --total-num-update 40000 --dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 --max-tokens 1024 --update-freq 2 --log-format simple --log-interval 2 --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
      --layernorm-embedding  --ddp-backend no_c10d'''
             ),
        ),  # noqa: E501
        'generate_kwargs': args_str_to_dict(
            f'''--task translation_from_pretrained_bart --source_lang {source_lang} --target-lang {target_lang} --batch-size 32 --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN'''  # noqa: E501
        ),
        # 'evaluate_kwargs': get_evaluate_kwargs(language),
    }

    return kwargs


# %%
dataset = 'uts_nl_query-9fcb6f786a1339d290dde06e16935402_db-9fcb6f786a1339d290dde06e16935402_topk-8_nprobe-16_density-0.6_distance-0.05_filter_ne-False_levenshtein-0.2_simplicity-0.0'

kwargs = get_mbart_kwargs(dataset=dataset, language='nl', use_access=True)
kwargs['train_kwargs']['ngpus'] = 1  # Set this from 8 to 1 for local training
kwargs['train_kwargs']['max_tokens'] = 512  # Lower this number to prevent OOM

#kwargs['train_kwargs']['optimizer'] = 'cpu_adam'
#kwargs['train_kwargs']['cpu-offload'] = True
#kwargs['train_kwargs']['ddp-backend'] = 'fully_sharded'
#kwargs['train_kwargs']['memory-efficient-fp16'] = True
kwargs['train_kwargs']['warmup_updates'] = 1
kwargs['train_kwargs']['total-num-update'] = 2
kwargs['train_kwargs']['max-update'] = 2


@clear_cuda_cache
def fairseq_train(
    preprocessed_dir,
    exp_dir,
    ngpus=1,
    # Batch size across all gpus (taking update freq into account)
    batch_size=8192,
    max_sentences=64,  # Max sentences per GPU
    arch='transformer',
    save_interval_updates=100,
    max_update=50000,
    lr=0.001,
    warmup_updates=4000,
    dropout=0.1,
    lr_scheduler='inverse_sqrt',
    criterion='label_smoothed_cross_entropy',
    seed=None,
    fp16=True,
    **kwargs,
):
    with log_std_streams(exp_dir / 'fairseq_train.stdout'):
        exp_dir = Path(exp_dir)
        preprocessed_dir = Path(preprocessed_dir)
        exp_dir.mkdir(exist_ok=True, parents=True)
        # Copy dictionaries to exp_dir for generation
        for dict_path in preprocessed_dir.glob('dict.*.txt'):
            shutil.copy(dict_path, exp_dir)
        checkpoints_dir = exp_dir / 'checkpoints'
        total_real_batch_size = max_sentences * ngpus
        update_freq = int(round(batch_size / total_real_batch_size, 0))
        if seed is None:
            seed = random.randint(0, 1000)
        distributed_port = random.randint(10000, 20000)
        #lr_scheduler = 'inverse_sqrt'
        args = f'''
        {preprocessed_dir} --task translation --source-lang complex --target-lang simple --save-dir {checkpoints_dir}
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0
        --criterion {criterion} --label-smoothing 0.1
        --lr-scheduler {lr_scheduler} --lr {lr} --warmup-updates {warmup_updates} --update-freq {update_freq}
        --arch {arch} --dropout {dropout} --weight-decay 0.0 --clip-norm 0.1 --share-all-embeddings
        --no-epoch-checkpoints --save-interval 999999 --validate-interval 999999
        --max-update {max_update} --save-interval-updates {save_interval_updates} --keep-interval-updates 1 --patience 10
        --batch-size {max_sentences} --seed {seed}
        --distributed-world-size {ngpus} --distributed-port {distributed_port}
        '''
        if lr_scheduler == 'inverse_sqrt':
            args += '--warmup-init-lr 1e-07'
        if fp16:
            args += f' --fp16'
        # FIXME: if the kwargs are already present in the args string, they will appear twice but fairseq will take only the last one into account
        args += f' {args_dict_to_str(kwargs)}'
        args = remove_multiple_whitespaces(args.replace('\n', ' ')).strip(' ')
        # Recover lost quotes around adam betas
        args = re.sub(r'--adam-betas (\(0\.\d+, 0\.\d+\))',
                      r"--adam-betas '\1'", args)
        print(f'fairseq-train {args}')
        with mock_cli_args(shlex.split(args)):
            train.cli_main()


# %%

preprocessed_dir = './resources/datasets/fairseq_preprocessed_complex-simple'

dir_name = f'local_{int(time.time() * 1000)}'
exp_dir = Path('./experiments/fairseq/') / dir_name
train_kwargs = kwargs.get('train_kwargs', {})

print_running_time(fairseq_train)(
    preprocessed_dir, exp_dir=exp_dir, **train_kwargs)
