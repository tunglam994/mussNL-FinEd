# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mussNLFined.muss.fairseq.main import fairseq_train_and_evaluate_with_parametrization
from mussNLFined.muss.mining.training import get_bart_kwargs, get_score_rows, get_mbart_kwargs
from mussNLFined.muss.resources.prepare import prepare_wikilarge_detokenized, prepare_asset
from mussNLFined.muss.resources.datasets import create_smaller_dataset


# This dataset should exist in resources/datasets/ and contain the following files:
# train.complex, train.simple, valid.complex, valid.simple, test.complex, test.simple
prepare_wikilarge_detokenized()
prepare_asset()

#dataset = 'wikilarge_detokenized'
dataset = 'uts_en_query-9c9aa1cf05b77f6cd018a159bd9eaeb0_db-9c9aa1cf05b77f6cd018a159bd9eaeb0_topk-8_nprobe-16_density-0.6_distance-0.05_levenshtein-0.2_simplicity-0.0-wo_turkcorpus'

kwargs = get_mbart_kwargs(dataset=dataset, language='nl', use_access=True)
kwargs['train_kwargs']['ngpus'] = 1  # Set this from 8 to 1 for local training
kwargs['train_kwargs']['max_tokens'] = 512  # Lower this number to prevent OOM

#kwargs['train_kwargs']['optimizer'] = 'cpu-adam'
#kwargs['train_kwargs']['cpu-offload'] = True
#kwargs['train_kwargs']['ddp-backend'] = 'fully_sharded'
#kwargs['train_kwargs']['warmup_updates'] = 1
#kwargs['train_kwargs']['total-num-update'] = 2
#kwargs['train_kwargs']['max-update'] = 2


result = fairseq_train_and_evaluate_with_parametrization(**kwargs)
