from datasets import load_dataset
yelp_review_full = load_dataset("billsum")
print(yelp_review_full.column_names)
yelp_review_full.set_format(type='tensorflow', columns=['label', 'text'])
yelp_review_full.save_to_disk('./billsum')