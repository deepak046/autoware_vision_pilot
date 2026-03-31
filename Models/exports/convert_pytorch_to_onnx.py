#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
"""
Convert PyTorch models to ONNX format.

Supported models:
  SceneSeg, Scene3D, DomainSeg, AutoSpeed, EgoLanes, AutoSteer, EgoLanesLite

EgoLanesLite requires a YAML config (--config / -c) to reconstruct the
architecture from training settings.

Usage examples:

  # Legacy models
  python -m Models.exports.convert_pytorch_to_onnx -n EgoLanes -p ckpt.pth -o model.onnx

  # Lite models (need config)
  python -m Models.exports.convert_pytorch_to_onnx -n EgoLanesLite \\
      -c Models/config/EgoLanesLite.yaml \\
      -p runs/training/.../checkpoints/best.pth \\
      -o ego_lanes_lite.onnx

  # Convert ONNX to TensorRT engine (after export):
  #   trtexec --onnx=ego_lanes_lite.onnx --saveEngine=ego_lanes_lite.engine --fp16
"""

import torch
import onnx
from argparse import ArgumentParser
import sys
sys.path.append('..')
from Models.model_components.scene_seg_network import SceneSegNetwork
from Models.model_components.scene_3d_network import Scene3DNetwork
from Models.model_components.domain_seg_network import DomainSegNetwork
from Models.model_components.auto_speed_network import AutoSpeedNetwork
from Models.model_components.ego_lanes_network import EgoLanesNetwork
from Models.model_components.auto_steer_network import AutoSteerNetwork

LITE_MODELS = {"EgoLanesLite"}


def _build_lite_model(model_name, config_path, checkpoint_path, device):
    """Build a lite model from YAML config and load checkpoint weights."""
    from Models.data_utils.lite_models.helpers.training import load_yaml
    from Models.inference.ego_lanes_lite_infer import EgoLanesLiteInferModel

    if config_path is None:
        raise ValueError(
            f"{model_name} requires a YAML config file. "
            f"Pass it via --config / -c."
        )

    cfg = load_yaml(config_path)
    device_str = str(device)
    infer_wrapper = EgoLanesLiteInferModel(cfg, checkpoint_path, device=device_str)

    aug_cfg = cfg.get("dataset", {}).get("augmentations", {}).get("rescaling", {})
    h = aug_cfg.get("height", 416)
    w = aug_cfg.get("width", 800)
    input_shape = (1, 3, h, w)

    return infer_wrapper.model, input_shape


def main():

    parser = ArgumentParser()
    parser.add_argument("-n", "--name", dest="network_name", required=True,
                        help="name of the network to export: SceneSeg | Scene3D | "
                             "DomainSeg | AutoSpeed | EgoLanes | AutoSteer | EgoLanesLite")

    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", required=True,
                        help="path to pytorch checkpoint file to load model dict")

    parser.add_argument("-o", "--onnx_model_path", dest="onnx_model_path", required=True,
                        help="path to converted ONNX model, must include output file name with .onnx extension")

    parser.add_argument("-c", "--config", dest="config", default=None,
                        help="(required for lite models) path to YAML config used during training")

    args = parser.parse_args()

    model_name = args.network_name
    model_checkpoint_path = args.model_checkpoint_path
    onnx_model_path = args.onnx_model_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference')

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = None
    input_shape = None
    skip_legacy_load = False

    if model_name in LITE_MODELS:
        print(f'Processing {model_name} (lite model)')
        model, input_shape = _build_lite_model(
            model_name, args.config, model_checkpoint_path, device
        )
        skip_legacy_load = True

    elif model_name == 'SceneSeg':
        print('Processing SceneSeg Network')
        model = SceneSegNetwork()
    elif model_name == 'Scene3D':
        print('Processing Scene3D Network')
        sceneSegNetwork = SceneSegNetwork()
        model = Scene3DNetwork(sceneSegNetwork)
    elif model_name == 'DomainSeg':
        print('Processing DomainSeg Network')
        sceneSegNetwork = SceneSegNetwork()
        model = DomainSegNetwork(sceneSegNetwork)
    elif model_name == 'AutoSpeed':
        print('Processing AutoSpeed Network')
        autospeed_builder = AutoSpeedNetwork()
        model = autospeed_builder.build_model(version='n', num_classes=4)
    elif model_name == 'EgoLanes':
        print('Processing EgoLanes Network')
        model = EgoLanesNetwork()
    elif model_name == 'AutoSteer':
        print('Processing AutoSteer Network')
        model = AutoSteerNetwork()
    else:
        raise Exception(
            f"Unknown model name '{model_name}'. "
            f"Supported: SceneSeg, Scene3D, DomainSeg, AutoSpeed, EgoLanes, AutoSteer, EgoLanesLite"
        )

    # ------------------------------------------------------------------
    # Load checkpoint (legacy models only; lite models already loaded)
    # ------------------------------------------------------------------
    if not skip_legacy_load:
        print('Loading Network')
        if len(model_checkpoint_path) > 0:
            checkpoint = torch.load(model_checkpoint_path, weights_only=False, map_location=device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                if hasattr(checkpoint['model'], 'state_dict'):
                    model.load_state_dict(checkpoint['model'].state_dict())
                else:
                    model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise ValueError('No path to checkpoint file provided')
        model = model.to(device)
        model = model.eval()

    # ------------------------------------------------------------------
    # Determine input shape (legacy models)
    # ------------------------------------------------------------------
    if input_shape is None:
        if model_name == 'AutoSpeed':
            input_shape = (1, 3, 640, 640)
        elif model_name == 'AutoSteer':
            input_shape = (1, 6, 80, 160)
        else:
            input_shape = (1, 3, 320, 640)

    print(f'Input shape: {input_shape}')
    input_data = torch.randn(input_shape, device=device)

    # ------------------------------------------------------------------
    # Test forward pass
    # ------------------------------------------------------------------
    print('Testing inference')
    with torch.no_grad():
        output = model(input_data)
    if isinstance(output, torch.Tensor):
        print(f'Output shape: {tuple(output.shape)}')
    else:
        print(f'Output type: {type(output)}')

    # ------------------------------------------------------------------
    # Export to ONNX
    # ------------------------------------------------------------------
    print('Converting model to ONNX at FP32 and exporting')
    torch.onnx.export(
        model,
        input_data,
        onnx_model_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )

    # Validate
    ONNX_network = onnx.load(onnx_model_path)
    onnx.checker.check_model(ONNX_network)
    print(f'Checks passed - export complete: {onnx_model_path}')

if __name__ == '__main__':
    main()
# %%