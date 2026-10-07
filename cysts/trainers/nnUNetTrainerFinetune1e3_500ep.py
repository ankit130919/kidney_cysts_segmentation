"""Fine-tune trainer: lr 1e-3, 500 epochs, for the kidney-CROPPED dataset.

lr 1e-3 rather than nnU-Net's 1e-2 for the reason established on the liver run: a converged
checkpoint warm-started at full LR loses small-lesion sensitivity in the first epochs.

500 epochs as requested. Note this is 2x the 250 used on the uncropped Dataset793, where
specificity had started to degrade past ~250 while pseudo-Dice still crept up. The crop
changes that calculus: cyst prevalence per patch is far higher once the background thorax
and pelvis are gone, so each epoch carries more cyst signal and less opportunity to learn
"kidneys usually contain lesions" from context that is no longer in frame.
"""
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerFinetune1e3_500ep(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_lr = 1e-3
        self.num_epochs = 500
