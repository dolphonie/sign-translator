"""
LanguageModel wrapper class for pre-trained huggingface GPT2 transformer. Provides interface methods
for batch tokenization, forward passes, and access to token and position embedding layers.

Usage example:

m = LanguageModel("gpt2-medium")
hidden, past, attention = m.forward(*m.tokenize(["hello, my name", "what can I say"]))
next_hidden, past, attention = m.forward(*m.tokenize(["is", "except"], add_space=True), past,
attention)
"""

# Imports
from typing import List, Tuple, Union

import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2Model


class LanguageModel(nn.Module):
    def __init__(self, pretrained: str = "gpt2"):
        """
        :param pretrained: name of the pretrained model, from ("gpt2", "gpt2-medium",
        "gpt2-large", "gpt2-xl")
        """
        super(LanguageModel, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained(pretrained)

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                past_key_values: Union[None, Tuple[torch.Tensor, ...]] = None,
                past_attention_mask: Union[None, torch.LongTensor] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        :param input_ids: (batch_size, input_id_len) token indices
        :param attention_mask: (batch_size, input_id_len) 1 for tokens that should be attended
        to, 0 for padding, etc.
        :param past_key_values: Optional, num_layers-len tuple of tensors each with shape
                                (2, batch_size, num_heads, past_tokens_len, embed_size_per_head)
                                the cache output of a previous call to the language model
                            Note: tokens represented in past_key_values should NOT be included in
                            input_ids
        :param past_attention_mask: Optional, (batch_size, past_tokens_len) attention mask
                                    corresponding to past key values inputs
        :return: (last_hidden_state, key_values, new_attention_mask)
                 last_hidden_state: (batch_size, input_id_len, hidden_size) final hidden state
                 associated
                                    with each input_id
                 key_values: updated past_key_values
                 new_attention_mask: (batch_size, past_tokens_len + input_id_len) new past
                 attention mask
        """
        if past_key_values is not None:
            new_attention_mask = torch.cat([past_attention_mask, attention_mask], dim=1)
        else:
            new_attention_mask = attention_mask
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=new_attention_mask,
            past_key_values=past_key_values
        )
        last_hidden_state = model_output.last_hidden_state
        key_values = model_output.past_key_values
        return last_hidden_state, key_values, new_attention_mask

    def tokenize(self, inputs: List[str], add_space: bool = False) -> Tuple[
        torch.LongTensor, torch.LongTensor]:
        """
        :param inputs: list of strings to tokenize
        :param add_space: True if a space should be added to the strings so that the first token
        is considered
                          a new word, else False
        :return: (input_ids, attention_mask)
                 each item has shape (batch_size=len(inputs), seq_len) where seq_len is the max #
                 of tokens in any input
        """
        inputs = ["<|endoftext|>" + el for el in inputs]

        if add_space:
            inputs = [" " + x for x in inputs]

        tokens = self.tokenizer.batch_encode_plus(
            inputs,
            padding="longest",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return tokens["input_ids"], tokens["attention_mask"]

    def token_embedding_layer(self) -> nn.Embedding:
        return self.model.wte

    def token_embedding(self) -> torch.Tensor:
        """
        Returns the token embedding tensor of shape (vocab_size, hidden_size).
        """
        return self.model.wte.weight.data

    def position_embedding(self) -> torch.Tensor:
        """
        Returns the position embedding tensor of shape (max_seq_len, hidden_size).
        """
        return self.model.wpe.weight.data
