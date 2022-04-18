import time
start_time = time.time()

print("Start")

from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(5))

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")

import numpy as np
from datasets import load_metric

# metric = load_metric("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy="epoch",
                                  eval_accumulation_steps=1,
                                  per_device_train_batch_size=1,
                                  per_device_eval_batch_size=1,
                                  num_train_epochs=1)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    # num_train_epochs=1
    # compute_metrics=compute_metrics,
)

trainer.train()


print("Done")
print("--- %s seconds ---" % (time.time() - start_time))
