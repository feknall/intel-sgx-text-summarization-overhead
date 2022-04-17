from transformers import AutoConfig, AutoModelForPreTraining

# Download configuration from huggingface.co and cache.
config = AutoConfig.from_pretrained("bert-base-cased")
model = AutoModelForPreTraining.from_config(config)