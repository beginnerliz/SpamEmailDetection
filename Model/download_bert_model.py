# Creator Cui Liz
# Time 08/07/2024 22:30

from transformers import BertModel, BertTokenizer

model_name = "bert-base-uncased"

model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

model.save_pretrained("./bert-base-uncased")
tokenizer.save_pretrained("./bert-base-uncased")
