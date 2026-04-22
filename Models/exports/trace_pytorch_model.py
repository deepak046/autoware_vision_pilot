#!/usr/bin/python3
import torch
from argparse import ArgumentParser
from Models.model_components.scene_seg_network import SceneSegNetwork
from Models.model_components.scene_3d_network import Scene3DNetwork
from Models.model_components.domain_seg_network import DomainSegNetwork
from Models.model_components.ego_lanes_network import EgoLanesNetwork, UNetPlusPlusNetwork
from Models.inference.ego_lanes_lite_infer import EgoLanesLiteInferModel

##
## Example Usage: "python3 traced_script_module_save.py -n SceneSeg -p _checkpoint_file_.pth  -o _output_trace_file.pt"
##

def main(): 

    # Command line arguments
    parser = ArgumentParser()
        
    parser.add_argument("-n", "--name", dest="network_name", required=True, \
                        help="specify the name of the network which will be benchmarked")

    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", required=True, \
                        help="path to pytorch checkpoint file to load model dict")
    
    parser.add_argument("-o", "--output_pt_trace_filepath", dest="output_pt_trace_filepath", required=True, \
                        help="path to *.pt output trace file generated")

    parser.add_argument("-c", "--config", dest="config", default=None, \
                        help="(required for EgoLanesLite) path to YAML config used during training")
    
    args = parser.parse_args() 

    # Model name, saved model checkpoint path and traced model save path
    model_name = args.network_name
    model_checkpoint_path = args.model_checkpoint_path
    traced_model_save_path = args.output_pt_trace_filepath

    # Checking devices (GPU vs CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'INFO: Using {device} for inference.')
        
    # Instantiating Model and setting to evaluation mode
    model = 0
    input_shape = (1, 3, 320, 640)
    if(model_name == 'SceneSeg'):
        print('Processing SceneSeg Network')
        model = SceneSegNetwork()
    elif (model_name == 'Scene3D'):
        print('Processing Scene3D Network')
        sceneSegNetwork = SceneSegNetwork()
        model = Scene3DNetwork(sceneSegNetwork)
    elif (model_name == 'EgoLanes'):
        print('Processing EgoPath Network')
        sceneSegNetwork = SceneSegNetwork()
        # model = EgoLanesNetwork()
        # model.load_state_dict(torch.load(model_checkpoint_path, weights_only=True, map_location=device))
        model = UNetPlusPlusNetwork()
        model.load_state_dict(torch.load(model_checkpoint_path, weights_only=True, map_location=device))
    elif (model_name == 'DomainSeg'):
        print('Processing DomainSeg Network')
        sceneSegNetwork = SceneSegNetwork()
        model = DomainSegNetwork(sceneSegNetwork)
    elif (model_name == 'EgoLanesLite'):
        print('Processing EgoLanesLite Network')
        if args.config is None:
            raise ValueError('EgoLanesLite requires --config pointing to the training YAML file')

        from Models.data_utils.lite_models.helpers.training import load_yaml
        cfg = load_yaml(args.config)
        infer_wrapper = EgoLanesLiteInferModel(cfg, model_checkpoint_path, device=str(device))
        model = infer_wrapper.model

        aug_cfg = cfg.get("dataset", {}).get("augmentations", {}).get("rescaling", {})
        h = int(aug_cfg.get("height", 416))
        w = int(aug_cfg.get("width", 800))
        input_shape = (1, 3, h, w)
    else:
        raise Exception("Model name not specified correctly, please check")
    
    # Loading Pytorch checkpoint
    if model_name != 'EgoLanesLite':
        print('Loading Network')
        if(len(model_checkpoint_path) > 0):
                model.load_state_dict(torch.load \
                    (model_checkpoint_path, weights_only=True, map_location=device))
        else:
            raise ValueError('No path to checkpiont file provided in class initialization')
    model = model.to(device)
    model = model.eval()

    # Fake input data
    input_data = torch.randn(input_shape)
    input_data = input_data.to(device)

    # Torch Export
    # Run and Trace the model with input image
    print('Tracing model')
    traced_script_module = torch.jit.trace(model, input_data)
    traced_script_module.save(traced_model_save_path) 
    print("Torch Trace Export file generated successfully.")


if __name__ == '__main__':
    main()