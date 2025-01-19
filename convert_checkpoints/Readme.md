The script for converting checkpoint config:

```
python convert_checkpoint.py convert-config config.json --model mistral --src-fmt hf --tgt-fmt cs-2.4 --output-dir new_dir
```

The script for converting checkpoint:

```
python convert_checkpoint.py convert ../mistral_checkpoints/model_tensors.safetensors --model mistral --src-fmt hf --tgt-fmt cs-2.4 --output-dir ../mistral_checkpoints/new_dir --config ../mistral_checkpoints/config.json --debug
```

The script for running the modelzoo model using a random h5 file:

```
python ~/ws/monolith3/monolith/src/models/src/cerebras/modelzoo/models/nlp/gpt2/run.py CPU --mo
de eval --params small_run_config.yaml --model_dir test_weights --checkpoint_path /net/srinjoym-dev/srv/nfs/srinjoym-data/ws/monolith3/monolith/src/models/src/cerebras/modelzoo/tools/new_dir/safetensors2_to_cs-2.4.mdl
```


