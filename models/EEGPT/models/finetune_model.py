"""
EEGPT Model Definition
"""
import torch
import torch.nn as nn
from .eegpt_mcae_finetune import EEGPTClassifier, CHANNEL_DICT


class Model(nn.Module):
    """
    EEGPT Model for fine-tuning

    Input: [Batch, Channel, Time]
    Output: [Batch, num_classes]
    """

    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params

        # Number of classes
        self.num_classes = getattr(params, 'num_of_classes', 2)

        # Number of channels
        self.num_channels = getattr(params, 'num_channels', 64)

        # Patch configuration
        self.patch_size = getattr(params, 'patch_size', 64)
        self.patch_stride = getattr(params, 'patch_stride', 64)

        # Model parameters
        # Data loader already filters channels to EEGPT's model channel space,
        # so ch_names should only contain channels in CHANNEL_DICT.
        self.original_ch_names = getattr(params, 'ch_names', None)
        if self.original_ch_names is not None:
            self.use_channels_names = list(self.original_ch_names)
            self.channel_indices = None  # No runtime selection needed
        else:
            # Fallback: use first N from CHANNEL_DICT
            self.use_channels_names = list(CHANNEL_DICT)[:self.num_channels]
            self.channel_indices = None

        # Update num_channels to match
        self.num_channels = len(self.use_channels_names)

        # Time length (img_size[1])
        img_size = getattr(params, 'img_size', [self.num_channels, 1024])
        if isinstance(img_size, int):
            img_size = [self.num_channels, img_size]
        else:
            img_size = [self.num_channels, img_size[1] if len(img_size) > 1 else 1024]

        # Desired time length for temporal interpolation (0 to disable)
        desired_time_len = getattr(params, 'desired_time_len', 1024)

        # Other parameters
        use_chan_conv = getattr(params, 'use_chan_conv', False)
        use_mean_pooling = getattr(params, 'use_mean_pooling', True)
        use_predictor = getattr(params, 'use_predictor', False)
        use_out_proj = getattr(params, 'use_out_proj', False)
        use_freeze_encoder = getattr(params, 'use_freeze_encoder', False)
        use_freeze_reconstructor = getattr(params, 'use_freeze_reconstructor', False)

        # Model
        self.backbone = EEGPTClassifier(
            num_classes=self.num_classes,
            in_channels=self.num_channels,
            img_size=img_size,
            patch_stride=self.patch_stride,
            use_channels_names=self.use_channels_names,
            channel_indices=self.channel_indices,
            use_mean_pooling=use_mean_pooling,
            use_chan_conv=use_chan_conv,
            use_predictor=use_predictor,
            use_out_proj=use_out_proj,
            use_freeze_encoder=use_freeze_encoder,
            use_freeze_reconstructor=use_freeze_reconstructor,
            desired_time_len=desired_time_len,
        )

        # Load pretrained weights
        if getattr(params, 'use_pretrained_weights', False):
            foundation_dir = getattr(params, 'foundation_dir', './models/Weights/EEGPT/eegpt_mcae_58chs_4s_large4E.ckpt')
            map_location = getattr(params, 'device_obj', torch.device('cpu'))
            try:
                checkpoint = torch.load(foundation_dir, map_location=map_location, weights_only=True)
            except Exception:
                # Fallback to weights_only=False for checkpoints with custom classes
                checkpoint = torch.load(foundation_dir, map_location=map_location, weights_only=False)
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                self.backbone.load_state_dict(checkpoint['model'], strict=False)
            elif 'state_dict' in checkpoint:
                self.backbone.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.backbone.load_state_dict(checkpoint, strict=False)

        if getattr(params, 'frozen', False):
            self.freeze_backbone()

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input [Batch, Channel, Time] or [Batch, Channel, Patch_Num, Patch_Size]

        Returns:
            Output [Batch, num_classes]
        """
        # Handle both (B, C, T) and (B, C, N, P) formats
        if len(x.shape) == 4:
            # Patched format: (B, C, N, P) -> (B, C, N*P)
            x = x.flatten(2)

        # Cast to float32 if input is half precision (for AMP compatibility)
        original_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()

        output = self.backbone(x)

        # Cast back to original dtype if needed
        if original_dtype == torch.float16:
            output = output.half()

        return output

    def extract_features(self, x):
        """Return backbone features for evaluator or downstream heads."""
        if len(x.shape) == 4:
            x = x.flatten(2)

        # Cast to float32 if input is half precision (for AMP compatibility)
        original_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.float()

        features = self.backbone.forward_features(x)

        # Cast back to original dtype if needed
        if original_dtype == torch.float16:
            features = features.half()

        return features

    def freeze_backbone(self):
        """Freeze backbone parameters for linear probing style fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Unfreeze the classifier head
        for param in self.backbone.head.parameters():
            param.requires_grad = True

    def get_param_groups(self):
        """
        Return optimizer parameter groups.
        lr_scale keeps previous behavior: classifier uses 5x learning rate.
        """
        backbone_params = []
        head_params = []

        for name, param in self.backbone.named_parameters():
            if name.startswith('head.'):
                head_params.append(param)
            else:
                backbone_params.append(param)

        encoder_lr_scale = getattr(self.params, 'encoder_lr_scale', 1.0)
        classifier_lr_scale = getattr(self.params, 'classifier_lr_scale', 5.0)
        return [
            {'params': backbone_params, 'lr_scale': encoder_lr_scale},
            {'params': head_params, 'lr_scale': classifier_lr_scale},
        ]
