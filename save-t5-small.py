from transformers import AutoTokenizer, AutoConfig, TFAutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.save_pretrained("./t5-small-tokenizer")

config = AutoConfig.from_pretrained("t5-small")
config.save_pretrained("./t5-small-config")

model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-small")
model.save_pretrained("./t5-small")