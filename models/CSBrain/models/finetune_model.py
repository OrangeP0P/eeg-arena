"""
CSBrain Model Wrapper - Adapts CSBrain to the unified framework interface
"""
import os
import sys
import torch
import torch.nn as nn

# Add current directory to path for importing CSBrain modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from .CSBrain import CSBrain as CSBrainBackbone
from .brain_regions import compute_brain_regions


class CSBrainFinetune(nn.Module):
    """
    Unified model contract:
    - forward(x) returns logits [B, num_classes]
    - extract_features(x) is optional but recommended

    CSBrain expects patched EEG input of shape (batch, channels, patch_num, patch_size)
    For base model: patch_size = 200

    CSBrain uses brain region information for brain-aware attention.
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

        # dataset-provided metadata (auto-filled by data loader)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)

        # Get channel names from params (provided by FinetuneDataLoader)
        ch_names = getattr(params, "ch_names", None)
        if ch_names is None:
            raise ValueError("ch_names is required for CSBrain. Make sure FinetuneDataLoader provides it.")

        # Compute brain regions
        brain_regions, sorted_indices = compute_brain_regions(ch_names)
        print(f"[CSBrain] Brain regions: {brain_regions}")
        print(f"[CSBrain] Sorted indices: {sorted_indices}")

        # CSBrain model hyperparameters
        dropout = getattr(params, "dropout", 0.0)
        d_model = 200
        self._d_model = d_model

        # Build CSBrain backbone
        self.backbone = CSBrainBackbone(
            in_dim=d_model,
            out_dim=d_model,
            d_model=d_model,
            dim_feedforward=800,
            seq_len=30,
            n_layer=12,
            nhead=8,
            brain_regions=brain_regions,
            sorted_indices=sorted_indices
        )

        # Load pretrained weights if enabled (using shape-matching filter
        # following the original CSBrain repo approach)
        if getattr(params, "use_pretrained_weights", False):
            foundation_dir = getattr(params, "foundation_dir", "./models/Weights/CSBrain")
            pretrained_file = os.path.join(foundation_dir, "CSBrain.pth")

            if os.path.exists(pretrained_file):
                checkpoint = torch.load(pretrained_file, map_location='cpu', weights_only=False)

                # Extract state_dict based on checkpoint format
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Handle checkpoint with 'module.' and 'backbone.' prefixes
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace("module.", "").replace("backbone.", "")
                    new_state_dict[new_key] = value

                # Shape-matching filter: only load weights with matching key AND shape
                model_state_dict = self.backbone.state_dict()
                matching_dict = {
                    k: v for k, v in new_state_dict.items()
                    if k in model_state_dict and v.size() == model_state_dict[k].size()
                }
                skipped = [
                    k for k in new_state_dict
                    if k in model_state_dict and new_state_dict[k].size() != model_state_dict[k].size()
                ]

                model_state_dict.update(matching_dict)
                self.backbone.load_state_dict(model_state_dict)

                print(f"[CSBrain] Loaded pretrained weights from {pretrained_file}")
                print(f"[CSBrain] Matched {len(matching_dict)}/{len(new_state_dict)} keys")
                if skipped:
                    print(f"[CSBrain] Skipped (shape mismatch): {skipped[:5]}... (total {len(skipped)})")
            else:
                print(f"[CSBrain] Warning: Pretrained weights not found at {pretrained_file}")

        # Replace proj_out with identity BEFORE freezing backbone
        self.backbone.proj_out = nn.Identity()

        # Freeze backbone if specified
        if getattr(params, "frozen", False):
            self.freeze_backbone()

        # Build classifier head
        # CSBrain outputs (batch, channels, patch_num, d_model)
        # Flatten to (batch, channels * patch_num * d_model)
        patch_num = getattr(params, "patch_num", 30)

        # Adaptive hidden size following original repo conventions
        classifier_hidden = max(4, patch_num) * d_model

        self.classifier = nn.Sequential(
            nn.Linear(num_channels * patch_num * d_model, classifier_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def extract_features(self, x):
        """
        Extract features from patched EEG input.

        Args:
            x: patched EEG data of shape (batch, channels, patch_num, patch_size)

        Returns:
            features of shape (batch, channels * patch_num * d_model)
        """
        bz, ch_num, seq_len, _ = x.shape
        feats = self.backbone(x)
        # Backbone output is (batch, channels, patch_num, d_model)
        feats = feats.contiguous().view(bz, ch_num * seq_len * self._d_model)
        return feats

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: patched EEG data of shape (batch, channels, patch_num, patch_size)

        Returns:
            logits of shape (batch, num_classes)
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self):
        """Freeze backbone parameters when --frozen is enabled."""
        for name, param in self.backbone.named_parameters():
            if 'proj_out' not in name:
                param.requires_grad = False
        print("[CSBrain] Backbone frozen (except projection layer)")

    def get_param_groups(self):
        """
        Optional API for optimizer parameter grouping.
        lr_scale is read from params (encoder_lr_scale, classifier_lr_scale).
        """
        backbone_params = []
        head_params = []

        for name, param in self.backbone.named_parameters():
            if 'proj_out' in name or 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        # Add classifier params
        for param in self.classifier.parameters():
            head_params.append(param)

        encoder_lr_scale = getattr(self.params, 'encoder_lr_scale', 1.0)
        classifier_lr_scale = getattr(self.params, 'classifier_lr_scale', 10.0)
        return [
            {"params": backbone_params, "lr_scale": encoder_lr_scale},
            {"params": head_params, "lr_scale": classifier_lr_scale},
        ]
