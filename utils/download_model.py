# Created by Patrick Kao
from transformers import GPT2Tokenizer, GPT2Model

from model.language_model import LanguageModel

pretrained = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(pretrained)
model = GPT2Model.from_pretrained(pretrained)
model.save_pretrained("pretrain_models/gpt2_model")
tokenizer.save_pretrained("pretrain_models/gpt2_tokenizer")