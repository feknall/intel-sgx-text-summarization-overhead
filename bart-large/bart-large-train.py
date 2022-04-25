import time
start_time = time.time()

print("Start")

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForPreTraining, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, TFAutoModelForSeq2SeqLM,\
    Seq2SeqTrainer
from transformers import AutoTokenizer
import numpy as np
from datasets import load_metric
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

from datasets import load_dataset
from datasets import load_from_disk

billsum = load_from_disk('billsum')

billsum = billsum.train_test_split(test_size=0.2)

print(billsum["train"][0])

tokenizer = AutoTokenizer.from_pretrained("knkarthick/MEETING_SUMMARY")

config = AutoConfig.from_pretrained("knkarthick/MEETING_SUMMARY")
model = TFAutoModelForSeq2SeqLM.from_config(config)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

prefix = "summarize: "


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_billsum = billsum.map(preprocess_function, batched=True)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")

tf_train_set = tokenized_billsum["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    shuffle=True,
    batch_size=1,
    collate_fn=data_collator,
)

tf_test_set = tokenized_billsum["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    shuffle=False,
    batch_size=1,
    collate_fn=data_collator,
)

from transformers import AdamWeightDecay

optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

model.compile(optimizer=optimizer)

model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=1)


print("Done")
print("--- %s seconds ---" % (time.time() - start_time))