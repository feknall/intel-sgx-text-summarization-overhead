from transformers import AutoTokenizer, AutoConfig, TFAutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("knkarthick/MEETING_SUMMARY")
tokenizer.save_pretrained("./t5-small-tokenizer")

config = AutoConfig.from_pretrained("knkarthick/MEETING_SUMMARY")
config.save_pretrained("./t5-small-config")

model = TFAutoModelForSeq2SeqLM.from_pretrained("knkarthick/MEETING_SUMMARY")
model.save_pretrained("./t5-small-model")