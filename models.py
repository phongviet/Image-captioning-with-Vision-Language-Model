import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ClipCapModel(nn.Module):
    def __init__(self, prefix_length=10, clip_dim=512, prefix_dim=768, gpt2_model='openai-community/gpt2'):
        super().__init__()
        self.prefix_length = prefix_length
        self.prefix_dim = prefix_dim

        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)

        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token

        self.clip_project = MLP(clip_dim, prefix_dim * 2, prefix_length * prefix_dim)

    def forward(self, image_features, input_ids=None, attention_mask=None, labels=None):
        batch_size = image_features.size(0)
        prefix_embeddings = self.clip_project(image_features).view(batch_size, self.prefix_length, self.prefix_dim)

        if input_ids is None:
            return prefix_embeddings

        embedding_cat = torch.cat((prefix_embeddings, self.gpt2.transformer.wte(input_ids)), dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, input_ids.size(1), device=input_ids.device)
        full_mask = torch.cat(
            (torch.ones(batch_size, self.prefix_length, device=input_ids.device), attention_mask),
            dim=1
        )

        full_labels = torch.cat((torch.full((batch_size, self.prefix_length), -100, dtype=torch.long, device=labels.device), labels), dim=1) if labels is not None else None

        return self.gpt2(inputs_embeds=embedding_cat, attention_mask=full_mask, labels=full_labels, return_dict=True)

    def generate(self, image_features):
        self.eval()

        batch_size = image_features.size(0)
        device = image_features.device

        with torch.no_grad():
            prefix_embeddings = self.forward(image_features)
            generated = self.gpt2.generate(
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.0,
                max_length=50,
                min_length=5,
                inputs_embeds=prefix_embeddings,
                pad_token_id=self.gpt2_tokenizer.pad_token_id,
                eos_token_id=self.gpt2_tokenizer.eos_token_id,
            )

            return [self.gpt2_tokenizer.decode(gen[self.prefix_length:], skip_special_tokens=True).split('.')[0].strip() + '.' for gen in generated]
