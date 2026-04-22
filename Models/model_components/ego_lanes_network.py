from .backbone import Backbone
from .backbone_feature_fusion import BackboneFeatureFusion
from .auto_steer_context import AutoSteerContext
from .ego_path_neck import EgoPathNeck
from .ego_lanes_head import EgoLanesHead

import segmentation_models_pytorch as smp

import torch.nn as nn

class EgoLanesNetwork(nn.Module):
    def __init__(self):
        super(EgoLanesNetwork, self).__init__()

        # Upstream blocks
        self.BEVBackbone = Backbone()

        # Feature Fusion
        self.BackboneFeatureFusion = BackboneFeatureFusion()

        # BEV Path Context
        self.AutoSteerContext = AutoSteerContext()

        # EgoPath Neck
        self.EgopathNeck = EgoPathNeck()

        # EgoPath Head
        self.EgoLanesHead = EgoLanesHead()
    

    def forward(self, image):
        features = self.BEVBackbone(image)
        fused_features = self.BackboneFeatureFusion(features)
        context = self.AutoSteerContext(fused_features)
        neck = self.EgopathNeck(context, features)
        ego_lanes = self.EgoLanesHead(neck)

        return ego_lanes

class SegFormerNetwork(nn.Module):
    def __init__(self):
        super(SegFormerNetwork, self).__init__()

        self.model = smp.Segformer(
            encoder_name="mit_b2",
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_segmentation_channels=256,
            in_channels=3,
            classes=3,
            # Keep logits here; trainer uses BCEWithLogitsLoss.
            activation=None,
            upsampling=4,
        )

    def forward(self, image):
        return self.model(image)

class UNetPlusPlusNetwork(nn.Module):
    def __init__(self):
        super(UNetPlusPlusNetwork, self).__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b0",
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type="scse",
            decoder_interpolation="bilinear",
            in_channels=3,
            classes=3,
            activation=None,
            upsampling=4,
        )

    def forward(self, image):
        return self.model(image)