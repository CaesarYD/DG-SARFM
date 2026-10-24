"""
Extract online encoder weights from a pretraining checkpoint for downstream fine-tuning.

Remove the encoder. prefix from state dict keys and save the weights in a new
checkpoint under the model field, excluding other modules and training state.
"""

import torch

if __name__ == '__main__':
    path = "/hdd0/xyd/project/mae/output_dir/new_cmae_param_model_output_dir/mae_38_imagenet_contrast_wo_cde_rde/checkpoint-99.pth"
    path_output = "/hdd0/xyd/project/mae/output_dir/new_cmae_param_model_output_dir/mae_38_imagenet_contrast_wo_cde_rde/refined_checkpoint-99.pth"
    checkpoint = torch.load(path, map_location='cpu')

    print("Load checkpoint from: %s" % path)
    checkpoint_model_cmae = checkpoint['model']
    checkpoint_model = {}
    for keys, values in checkpoint_model_cmae.items():
        split = keys.split('.', 1)
        if split[0] == 'encoder':
            checkpoint_model[split[1]] = values

    new = {'model': checkpoint_model}
    torch.save(new, path_output)
