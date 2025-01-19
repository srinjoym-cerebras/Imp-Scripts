import os
import json
from safetensors import safe_open

# List of `.safetensors` files
safetensors_file = "model.safetensors"

# Output JSON index file
index_file = "safetensors.index.json"

# Initialize the index structure
index_data = {
    "metadata": {
        "total_size": 0  # This will be calculated
    },
    "weight_map": {}
}

# Iterate over all `.safetensors` files
# for safetensors_file in safetensors_files:
file_size = os.path.getsize(safetensors_file)
index_data["metadata"]["total_size"] += file_size  # Accumulate total size

# Open the `.safetensors` file to extract tensor names
with safe_open(safetensors_file, framework="numpy") as f:
    tensor_names = list(f.keys())

# Map tensor names to the current `.safetensors` file
for tensor_name in tensor_names:
    index_data["weight_map"][tensor_name] = safetensors_file

# Save the index JSON file
with open(index_file, "w") as outfile:
    json.dump(index_data, outfile, indent=4)

print(f"Index file generated: {index_file}")
