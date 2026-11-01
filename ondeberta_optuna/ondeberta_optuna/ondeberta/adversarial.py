"""Adversarial training (FGM - Fast Gradient Method on the word embeddings).

Corresponds to the "Adversarial Training" block attached to the DeBERTa encoder
in the system diagram. Each step: normal forward/backward -> perturb the
embedding matrix along the gradient direction -> second forward/backward on the
perturbed input (gradients accumulate) -> restore the embeddings.
"""
import torch


class FGM:
    def __init__(self, model, epsilon: float = 1.0, emb_name: str = "word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and self.emb_name in name and p.grad is not None:
                self.backup[name] = p.data.clone()
                norm = torch.norm(p.grad)
                if torch.isfinite(norm) and norm != 0:
                    p.data.add_(self.epsilon * p.grad / norm)

    def restore(self):
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data = self.backup[name]
        self.backup = {}
