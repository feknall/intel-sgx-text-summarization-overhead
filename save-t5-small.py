from transformers import AutoTokenizer, AutoConfig, TFAutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-small2")
tokenizer.save_pretrained("./t5-small2-tokenizer")

config = AutoConfig.from_pretrained("t5-small2")
config.save_pretrained("./t5-small2-config")

model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-small2")
model.save_pretrained("./t5-small2")