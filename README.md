# Multilingual Unsupervised Sentence Simplification, Dutch Implementation

For people who have cognitive disabilities or are not native speakers of a language, long and complex sentences can be hard to comprehend. 
For this reason, social instances with a large reach, like governmental instances, have to address people in a simple manner. 
For a person who has a thorough understanding of a language, writing a ‘simple’ sentence can be challenging.

This problem can be solved with the use of deep learning models, where especially the rise of the transformer architecture caused a massive 
improvement in performance. The problem with deep learning models is the necessity for large quantities of labeled data.

This problem is countered in the paper [MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases](https://github.com/facebookresearch/muss), 
which, as the name suggests, can train a state of the art text simplification model with unlabelled data. To implement this strategy for 
the Dutch language, I forked the repository and made the necessary alterations. Furthermore, as the original code was trained on the 
Facebook supercomputer cluster, I made it work with my workstation with 32GB RAM and a RTX 2070 super 8GB.

Code and pretrained models to reproduce experiments in "MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases".

## Prerequisites

Linux with python 3.6 or above (not compatible with python 3.9 yet).

## Installing

```
git clone https://github.com/JohanBekker/mussNL.git
cd mussNL/
pip install -e .  # Install package
python -m spacy download nl_core_news_md  # Install required spacy models
```

Also, to use the NLTK Tokenizer, you should open a python terminal and execute the following commands

```python
import nltk
nltk.download()
```

After these commands, menu pops up. Navigate to download the package 'punkt', and you're good to go.

## How to use
Some scripts might still contain a few bugs, if you notice anything wrong, feel free to open an issue or submit a Pull Request.

### Simplify sentences from a file using pretrained models
```python
# English
python scripts/simplify.py scripts/examples.en --model-name muss_en_wikilarge_mined
# French
python scripts/simplify.py scripts/examples.fr --model-name muss_fr_mined
# Spanish
python scripts/simplify.py scripts/examples.es --model-name muss_es_mined
```

Pretrained models should be downloaded automatically, but you can also find them here:  
[muss_en_wikilarge_mined](https://dl.fbaipublicfiles.com/muss/muss_en_wikilarge_mined.tar.gz)  
[muss_en_mined](https://dl.fbaipublicfiles.com/muss/muss_en_mined.tar.gz)  
[muss_fr_mined](https://dl.fbaipublicfiles.com/muss/muss_fr_mined.tar.gz)  
[muss_es_mined](https://dl.fbaipublicfiles.com/muss/muss_es_mined.tar.gz)  

### Mine the data
```python
python scripts/mine_sequences.py
```

### Train the model
```python
python scripts/train_model.py
```

### Evaluate simplifications
Please head over to [EASSE](https://github.com/feralvam/easse/) for Sentence Simplification evaluation.


## License

The MUSS license is CC-BY-NC. See the [LICENSE](LICENSE) file for more details.

## Authors

* **Louis Martin** ([louismartincs@gmail.com](mailto:louismartincs@gmail.com))


## Citation

If you use MUSS in your research, please cite [MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases](https://arxiv.org/abs/2005.00352)

```
@article{martin2021muss,
  title={MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases},
  author={Martin, Louis and Fan, Angela and de la Clergerie, {\'E}ric and Bordes, Antoine and Sagot, Beno{\^\i}t},
  journal={arXiv preprint arXiv:2005.00352},
  year={2021}
}
```
