"""
Shared model definition — imported by BOTH train.py and predict.py
so the architecture always matches the saved weights.

NOTE: inplace=False on every ReLU. Inplace ReLU silently breaks Grad-CAM
(autograd refuses to differentiate through a tensor that was modified in
place). ReLU has no parameters, so this does NOT change the state_dict —
your existing brain_tumor_detector.pt loads fine. No retraining needed.
"""
import torch
import torch.nn as nn

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary Tumor"}
N_CLASSES = 4
IMG_SIZE = 128


def conv_block(in_ch, out_ch, dropout):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),   # 0
        nn.BatchNorm2d(out_ch),                               # 1
        nn.ReLU(inplace=False),                               # 2
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),  # 3
        nn.BatchNorm2d(out_ch),                               # 4
        nn.ReLU(inplace=False),                               # 5  <- Grad-CAM taps this
        nn.MaxPool2d(2),                                      # 6
        nn.Dropout2d(dropout),                                # 7
    )


class BrainTumorCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.block1 = conv_block(1,   32,  0.20)   # 128 -> 64
        self.block2 = conv_block(32,  64,  0.25)   # 64  -> 32
        self.block3 = conv_block(64,  128, 0.30)   # 32  -> 16
        self.block4 = conv_block(128, 256, 0.30)   # 16  -> 8

        self.features = nn.Sequential(
            self.block1, self.block2, self.block3, self.block4
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)

    def gradcam_target_layer(self):
        """
        Post-ReLU activations of the last conv block (256ch, 8x8).
        Index 5 = the second ReLU. Index 4 is BatchNorm, index 3 is the conv —
        standard Grad-CAM taps the activations AFTER the final ReLU.
        """
        return self.block4[5]
