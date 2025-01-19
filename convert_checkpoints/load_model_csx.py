# import cerebras.modelzoo.models.nlp.mistral.model as M
# import cerebras.modelzoo.models.nlp.gpt2.model as G
import yaml
# import pprint
# from cerebras.modelzoo.common.run_utils import run

import cerebras.pytorch as cstorch


mdl_file = 'new_dir/safetensors_to_cs-2.4.mdl'
config_file = 'new_dir/config_to_cs-2.4.yaml'

# with open(config_file, 'r') as file:
#     data = yaml.safe_load(file)


# pprint.pprint(data)
# print(dir(M))


# model_config = data['model']

# gpt_config= G.GPT2LMHeadModelConfig(**data)
# mistral_config = M.MistralModelConfig(**data)
# mistral_config = M.MistralModelConfig(**model_config)
# pprint.pprint(mistral_config.model_dump())
# pprint.pprint(gpt_config.model_dump())

# model = MistralModel(mistral_config)

# print(dir(Mistral))

model = cstorch.load(mdl_file)
# breakpoint()

for key in model['model'].keys():
    print(f'{key}: ', model['model'][key].shape)
# print(dir(model))
# python load_model_csx.py CPU --params run_config.yaml --mode eval --checkpoint_path new_dir/safetensors_to_cs-2.4.mdl
# # PYTHONPATH=~/ws/modelzoo-internal/modelzoo/src python   ~/ws/monolith3/monolith/src/models/src/cerebras/modelzoo/models/nlp/gpt2/run.py CPU \
    # --mode eval \
    # --params small_run_config.yaml \
    # --model_dir test_weights \
    # --checkpoint_path /net/srinjoym-dev/srv/nfs/srinjoym-data/ws/monolith3/monolith/src/models/src/cerebras/modelzoo/tools/new_dir/safetensors2_to_cs-2.4.mdl

# run()