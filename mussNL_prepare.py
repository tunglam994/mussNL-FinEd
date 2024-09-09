# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:15:05 2022

@author: johan
"""

import shutil
import random
import shlex
from pathlib import Path
import re

from mussNLFinEd.muss.mining.training import get_bart_kwargs, get_score_rows, get_mbart_kwargs
from mussNLFinEd.muss.utils.training import clear_cuda_cache

from mussNLFinEd.muss.fairseq.main import check_dataset, check_and_resolve_args, prepare_exp_dir
from mussNLFinEd.muss.fairseq.base import fairseq_preprocess, get_fairseq_exp_dir
# from mussNLFinEd.muss.fairseq.base import fairseq_train,
from mussNLFinEd.muss.preprocessors import get_preprocessors
from mussNLFinEd.muss.resources.datasets import create_preprocessed_dataset
from mussNLFinEd.muss.resources.paths import get_data_filepath, get_dataset_dir

from mussNLFinEd.muss.utils.helpers import log_std_streams, mock_cli_args, print_running_time
from fairseq_cli import preprocess, train, generate

from mussNLFinEd.muss.utils.helpers import (
    log_std_streams,
    lock_directory,
    create_directory_or_skip,
    yield_lines,
    write_lines,
    mock_cli_args,
    create_temp_dir,
    mute,
    args_dict_to_str,
)

from mussNLFinEd.muss.text import remove_multiple_whitespaces
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

# %%


def prepare_preprocessors_datasets(dataset, **kwargs):
    # check_dataset(dataset)
    kwargs = check_and_resolve_args(kwargs)
    exp_dir = prepare_exp_dir()
    preprocessors_kwargs = kwargs.get('preprocessors_kwargs', {})
    preprocessors = get_preprocessors(preprocessors_kwargs)
    if len(preprocessors) > 0:
        dataset = create_preprocessed_dataset(dataset, preprocessors, n_jobs=8)
        dataset_dir = get_dataset_dir(dataset)
        shutil.copy(dataset_dir / 'preprocessors.pickle', exp_dir)
        if hasattr(preprocessors[-1], 'copy_sentencepiece_files_to_dir'):
            preprocessors[-1].copy_sentencepiece_files_to_dir(dataset_dir)
    model_symlink_path = exp_dir / 'model.pt'
    if not model_symlink_path.exists():
        model_symlink_path.symlink_to('checkpoints/checkpoint_best.pt')
    preprocessed_dir = fairseq_preprocess(
        dataset, **kwargs.get('preprocess_kwargs', {}))
    train_kwargs = kwargs.get('train_kwargs', {})

    return preprocessed_dir, exp_dir, train_kwargs


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


preprocessed_dir, exp_dir, train_kwargs = print_running_time(
    prepare_preprocessors_datasets)(**kwargs)

# %%


# print_running_time(fairseq_train)(
#    preprocessed_dir, exp_dir=exp_dir, **train_kwargs)
