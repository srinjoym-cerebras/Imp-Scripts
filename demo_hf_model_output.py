from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from safetensors.torch import load_file
import torch
import json

hf_dir = ''
cstorch_dir = ''
model_dir = '/net/srinjoym-dev/srv/nfs/srinjoym-data/ws/modelzoo-internal/modelzoo/src/cerebras/modelzoo/mistral_checkpoints/ml2_new_draft_converted'
# model_weights_path = 'model_dir/model_renamed.safetensors'
# model_weights_path = '/net/srinjoym-dev/srv/nfs/srinjoym-data/ws/monolith3/monolith/src/models/src/cerebras/modelzoo/mistral_checkpoints/model_dir/new_renamed_model.safetensors'
# config_file_path = '/net/srinjoym-dev/srv/nfs/srinjoym-data/ws/monolith3/monolith/src/models/src/cerebras/modelzoo/mistral_checkpoints/model_dir/config.json'

# model_weights_path = '~/ws/modelzoo-internal/modelzoo/src/cerebras/modelzoo/mistral_checkpoints/ml2_new_draft_converted'




random_strings = []


def dump_hf_tensors():
    # config = AutoConfig.from_pretrained(config_file_path)
    # model = AutoModelForCausalLM.from_config(config)
    model = AutoModelForCausalLM.from_pretrained(model_dir, use_safetensors=True, torch_dtype=torch.bfloat16)
    print(model.config)    

    # state_dict = load_file(model_weights_path)
    # for key, weights in state_dict.items():
    #     print(f"{key}: {weights.shape}")

    # model.load_state_dict(state_dict, strict=False)

    # tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-Large-Instruct-2411', token=HF_TOKEN)
    # # # if model.config.pad_token_id is None:
    # # #     tokenizer.pad_token = tokenizer.eos_token
    # # #     model.config.pad_token_id = tokenizer.eos_token_id

    # if tokenizer.pad_token is None:
    #     print('XXXXXXXX No padding token in tokenizer')
    #     tokenizer.pad_token = tokenizer.eos_token


    # messages = [{'role': 'user', 'content': 'Explain Attention Mechanism'}, {'role': 'assistant', 'content': 'You are an AI assistant which helps humans in learning about new stuff'}]
    
    # inputs = tokenizer.apply_chat_template(messages, return_tensors='pt', padding=True, truncation=True, max_length = 128, return_dict=True)

    # file = '/net/srinjoym-dev/srv/nfs/srinjoym-data/ws/monolith3/monolith/src/models/src/cerebras/modelzoo/mistral_checkpoints/demo_file.jsonl'

    # texts = []

    # with open(file, 'r') as f:
    #     for line in f:
    #         data = json.loads(line)
    #         texts.append(data['content'])
    #         break

    x = torch.tensor([[1, 415, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 3914, 28723, 1, 3951, 14773, 10895, 349, 11029, 3864, 1287, 17909, 28723, 1, 985, 6112, 28713, 28443, 13436, 28713, 486, 272, 427, 1029, 431, 28723, 1, 415, 7296, 297, 12567, 22361, 11464, 297, 272, 10835, 28723, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]], dtype=torch.int)

    # inputs = tokenizer(texts, padding='max_length', max_length =128, truncation=True, return_tensors='pt')
    # print(inputs['input_ids'].shape)
    

    # inputs = torch.randint(0, 32768, (2, 100))
    attention_mask = torch.ones(x.shape)

    # output = model(input_ids=inputs, attention_mask=attention_mask, output_hidden_states = True)
    max_length=200
    # output = model.generate(x, max_length=max_length, output_hidden_states=True)
    output = model(input_ids=x, attention_mask=attention_mask, output_hidden_states=True)
    # output = model(**inputs, output_hidden_states = True)

    # hidden_states = output.hidden_states

    breakpoint()

    # for i, layer_output in enumerate(hidden_states):
    #     print(f'Layer_{i}: {layer_output.shape}')
    #     print(f'{layer_output}')
        # np.save(f'hf_layer_{i}_output.npy', layer_output.numpy())
    
    # breakpoint()


# def dump_cstorch_tensors():
    
#     pass


# def compare():
#     pass


def main():
    dump_hf_tensors()


if __name__ == "__main__":
    main()
