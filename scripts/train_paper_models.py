# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy

from mussNLFined.muss.utils.helpers import print_running_time
from mussNLFined.muss.utils.submitit import get_executor, print_job_id
from mussNLFined.muss.utils.training import print_function_name, print_args, print_result
from mussNLFined.muss.fairseq.main import fairseq_train_and_evaluate_with_parametrization
from mussNLFined.muss.fairseq.base import get_fairseq_exp_dir
from mussNLFined.muss.mining.training import (
    get_transformer_kwargs,
    get_bart_kwargs,
    get_mbart_kwargs,
    get_score_rows,
    get_all_baseline_rows,
)
from mussNLFined.muss.resources.datasets import mix_datasets, create_smaller_dataset


# =============================================================================
# def get_mean_confidence_interval(data, confidence=0.95):
#     data = np.array(data)
#     a = 1.0 * data
#     a = a[~np.isnan(a)]
#     n = len(a)
#     se = scipy.stats.sem(a)
#     h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
#     return h
# =============================================================================

# =============================================================================
# def get_formatted_mean_and_confidence_interval(array, confidence=0.95):
#     array = np.array(array)
#     mean = np.mean(array)
#     confidence_interval = get_mean_confidence_interval(array, confidence=confidence)
#     return f"{mean:.2f}±{confidence_interval:.2f} ({array.size})"
# =============================================================================


print('WARNING: This script probably won\'t work flawlessly out of the box, but it should be easy to understand how to make it work for your use case.')
#wikilarge = 'wikilarge_detokenized-wo_turkcorpus'
uts_nl_1bq_paraphrases = 'uts_en_query-9c9aa1cf05b77f6cd018a159bd9eaeb0_db-9c9aa1cf05b77f6cd018a159bd9eaeb0_topk-8_nprobe-16_density-0.6_distance-0.05_levenshtein-0.2_simplicity-0.0-wo_turkcorpus'  # noqa: E501

kwargs_dict = {
    # French table
    'mbart_access_uts_nl_1bq_paraphrases_wo_supervised': get_mbart_kwargs(
        dataset=uts_nl_1bq_paraphrases, language='nl', use_access=True
    ),
}

# Training model
for name, kwargs in tqdm(kwargs_dict.items()):
    fairseq_train_and_evaluate_with_parametrization(**kwargs)

# =============================================================================
# jobs_dict = defaultdict(list)
# for exp_name, kwargs in tqdm(kwargs_dict.items()):
#     executor = get_executor(
#         cluster='local',
#         slurm_partition='priority',
#         submit_decorators=[print_function_name, print_args, print_job_id, print_result, print_running_time],
#         timeout_min=2 * 24 * 60,
#         gpus_per_node=kwargs['train_kwargs']['ngpus'],
#         nodes=1,
#         slurm_constraint='volta32gb',
#         name=exp_name,
#     )
#     for i in range(5):
#         job = executor.submit(fairseq_train_and_evaluate_with_parametrization, **kwargs)
#         jobs_dict[exp_name].append(job)
# [job.result() for jobs in jobs_dict.values() for job in jobs]
# =============================================================================

# Evaluation
# =============================================================================
# table = []
# for exp_name, jobs in tqdm(jobs_dict.items()):
#     for job in jobs:
#         exp_dir = get_fairseq_exp_dir(job.job_id)
#         kwargs = job.submission().kwargs
#         table.extend(get_score_rows(exp_dir, kwargs, additional_fields={'exp_name': exp_name, 'job_id': job.job_id}))
# table.extend(print_running_time(get_all_baseline_rows)())
# df_scores = pd.DataFrame(table)
# =============================================================================

# =============================================================================
# def mean(arr):
#     if len(arr) not in [8, 10]:  # Hack for Reference rows
#         arr = arr[-5:]
#     return get_formatted_mean_and_confidence_interval(arr)
# =============================================================================

# =============================================================================
# pd.set_option('display.max_rows', 300)
# pd.set_option('display.max_colwidth', 100)
# print(
#     df_scores.groupby(['language', 'dataset', 'phase', 'exp_name'])
#     .agg([mean])[['sari', 'bleu', 'fkgl']]
#     .sort_values(by=['language', 'dataset', 'phase', ('sari', 'mean')])
# )
# =============================================================================
