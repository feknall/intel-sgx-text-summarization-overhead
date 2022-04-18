import time
start_time = time.time()

print("Start")

from datasets import load_from_disk

dataset = load_from_disk("yelp_review_full2")
# print(f"{dataset.format}********")
print(dataset["train"][100])

from transformers import AutoTokenizer, TFTrainer, DataCollatorForSeq2Seq

tokenizer = AutoTokenizer.from_pretrained("./bert-base-cased-tokenizer")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained("./bert-base-cased", num_labels=5, from_pt=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")

small_train_dataset = tokenized_datasets["train"].to_tf_dataset(columns=['label', 'text'],
    shuffle=True,
    batch_size=1,
    collate_fn=data_collator).shuffle(seed=42).select(range(10))
# small_eval_dataset = tokenized_datasets["test"].to_tf_dataset().shuffle(seed=42).select(range(10))


import numpy as np
# from datasets import load_metric

# metric = load_metric("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer
from transformers import TFTrainingArguments
training_args = TFTrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch",
                                  eval_accumulation_steps=1,
                                  per_device_train_batch_size=1, per_device_eval_batch_size=1)

trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    # eval_dataset=small_eval_dataset,
    # compute_metrics=compute_metrics,
)

trainer.train()

print("Done")
print("--- %s seconds ---" % (time.time() - start_time))
