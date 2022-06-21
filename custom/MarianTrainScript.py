from transformers import (MarianTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, AutoConfig, AutoModelForSeq2SeqLM)

from datasets import load_from_disk, load_metric

import numpy as np

tokenizer = MarianTokenizer.from_pretrained("tokenizer/")
config = AutoConfig.from_pretrained("Helsinki-NLP/opus-mt-nl-en")
cache = "models/pretrained/"
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-nl-en", config=config, cache_dir=cache)

batch_size = 6
gr_accum_steps = 21#16
model_name = "MarianMTModel"
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-MussNL",
    #evaluation_strategy = "epoch",
    evaluation_strategy = "steps",
    learning_rate=3e-05,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gr_accum_steps,
    #gradient_checkpointing=True,
    # label_smoothing_factor=0.2,
    #optim="adafactor",
    lr_scheduler_type='polynomial',
    warmup_steps=2500,
    fp16=True,
    weight_decay=0.01,
    save_total_limit=3,
    eval_steps=10000,#2000,
    max_steps=40000,
    # num_train_epochs=2,
    predict_with_generate=True,
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model)

metric = load_metric("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # if data_args.ignore_pad_token_for_loss:
    #     # Replace -100 in the labels as we can't decode them.
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


data_dir = "data/tokenized/"
dataset = load_from_disk(data_dir)
train_dataset = dataset["train"]
eval_dataset = dataset["valid"]

trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

trainer.train()
trainer.save_model()

results = {}
max_length = 512

num_beams = 6

metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

metrics_test = trainer.evaluate(eval_dataset=dataset["test"], max_length=max_length, num_beams=num_beams, metric_key_prefix="test")
trainer.log_metrics("test", metrics_test)
trainer.save_metrics("test", metrics_test)