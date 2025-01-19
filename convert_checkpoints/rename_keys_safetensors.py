from safetensors.torch import save_file, load_file
import re
# File paths
#input_file = "model_dir/model.safetensors"
#output_file = "model_dir/model_renamed.safetensors"

input_file = 'model_dir/new_model.safetensors'
output_file = 'model_dir/new_renamed_model.safetensors'


replacement_rules = [
    ('tok_embeddings', 'embed_tokens'),
    # (r'\.layers\.', '.tranformer_decoder.layers.'),
    (r'\.wk\.', '.k_proj.'),
    (r'\.wo\.', '.o_proj.'),
    (r'\.wq\.', '.q_proj.'),
    (r'\.wv\.', '.v_proj.'),
    (r'\.attention\.', '.self_attn.'),
    (r'\.feed_forward\.', '.mlp.'),
    (r'\.w2\.', '.down_proj.'),
    (r'\.w1\.', '.gate_proj.'),
    (r'\.w3\.', '.up_proj.'),
    (r'\.ffn_norm\.', '.post_attention_layernorm.'),
    (r'\.attention_norm\.', '.input_layernorm.'),
    ('output.', 'lm_head.'),

]

# Define the renaming function
def rename_key(key):
    # Example rules for renaming:
    # 1. Replace "attention" with "attn"
#    key = key.replace("attention", "attn")
    # 2. Add a prefix to all keys
#    key = "renamed_" + key
    # 3. Replace dots with underscores
#    key = key.replace(".", "_")
#    return key
    
    new_key = key
    for pattern, replacement in replacement_rules:
        #if new_key.endswith('qscale_act') or new_key.endswith('qscale_weight'):
        #    continue
        new_key = re.sub(pattern, replacement, new_key)
    
    if not new_key.startswith('lm_head'):
        new_key = 'model.'+new_key    
    return new_key


# Load the safetensors file
weights = load_file(input_file)

# Rename keys using the rules
renamed_weights = {}

for key, value in weights.items():
    if key.endswith('qscale_weight') or key.endswith('qscale_act'):
        continue
    #else:
    renamed_weights[rename_key(key)]=value


# renamed_weights = {rename_key(key): value for key, value in weights.items()}

# Save the renamed weights back to a new safetensors file
save_file(renamed_weights, output_file)

print(f"Renamed safetensors file saved to {output_file}")
