from transformers import AutoTokenizer, AutoConfig, TFAutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-base")
tokenizer.save_pretrained("./t5-base-tokenizer")

config = AutoConfig.from_pretrained("t5-base")
config.save_pretrained("./t5-base-config")

model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-base")
model.save_pretrained("./t5-base-model")