from safetensors.torch import load_file

# Path to the .safetensors file
#safetensors_file_path = "model_dir/model_renamed.safetensors"
# safetensors_file_path = 'model_dir/new_model.safetensors'
safetensors_file_path = "model_dir/new_renamed_model.safetensors"
#safetensors_file_path = 'model_dir/model.safetensors'
# Load the file
print(f'the model loaded: {safetensors_file_path}')

state_dict = load_file(safetensors_file_path)

# Print all keys
print("Keys in the safetensor file:")
for key, weights in state_dict.items():
#    if key.endswith('qscale_weight') or key.endswith('qscale_act'):
#        print(f"{key}: {weights}")
#    else:
#        print(f"{key}: {weights.shape}, {type(weights)}")
    print(f'{key}')
