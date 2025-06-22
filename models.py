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

        self.clip_project = MLP(
            input_dim=clip_dim,
            hidden_dim=prefix_dim * 2,
            output_dim=prefix_length * prefix_dim
        )

    def get_dummy_token(self, batch_size, device):
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.long, device=device)

    def forward(self, image_features, input_ids=None, attention_mask=None, labels=None):
        batch_size = image_features.size(0)
        # MLP mapping
        prefix_projections = self.clip_project(image_features)
        prefix_embeddings = prefix_projections.view(batch_size, self.prefix_length, self.prefix_dim)

        if input_ids is not None:
            # Pass through GPT2 with prefix embeddings
            embedding_text = self.gpt2.transformer.wte(input_ids)
            embedding_cat = torch.cat((prefix_embeddings, embedding_text), dim=1)

            # Create attention mask including prefix
            if attention_mask is not None:
                prefix_mask = torch.ones(batch_size, self.prefix_length, device=attention_mask.device)
                full_mask = torch.cat((prefix_mask, attention_mask), dim=1)
            else:
                full_mask = None

            # Adjust labels to account for prefix
            if labels is not None:
                # Add -100 for prefix positions (to be ignored by loss calculation)
                prefix_labels = torch.full((batch_size, self.prefix_length), -100,
                                           dtype=torch.long, device=labels.device)
                full_labels = torch.cat((prefix_labels, labels), dim=1)
            else:
                full_labels = None

            outputs = self.gpt2(
                inputs_embeds=embedding_cat,
                attention_mask=full_mask,
                labels=full_labels,
                return_dict=True  # This ensures we get an object with .loss attribute
            )

            return outputs
        else:
            # Just return the prefix embeddings if no text input
            return prefix_embeddings

    def generate(self, image_features, max_length=20, temperature=0.7,
                 do_sample=True, top_p=0.9, num_beams=3):
        self.eval()
        batch_size = image_features.size(0)
        device = image_features.device

        with torch.no_grad():
            prefix_embeddings = self.forward(image_features)
            prefix_tokens = self.get_dummy_token(batch_size, device)

            generated = self.gpt2.generate(
                input_ids=prefix_tokens,
                inputs_embeds=prefix_embeddings,
                max_length=max_length + self.prefix_length,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                pad_token_id=self.gpt2_tokenizer.pad_token_id,
                eos_token_id=self.gpt2_tokenizer.eos_token_id,
                early_stopping=True,
                # Force the model to stop at the first period
                forced_eos_token_id=self.gpt2_tokenizer.encode('.')[0]
            )

            generated = generated[:, self.prefix_length:]

            captions = []
            for gen in generated:
                text = self.gpt2_tokenizer.decode(gen, skip_special_tokens=True)
                # Take only the first sentence
                if '.' in text:
                    text = text.split('.')[0] + '.'
                captions.append(text.strip())
            return captions