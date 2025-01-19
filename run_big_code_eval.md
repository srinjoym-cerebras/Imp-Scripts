Script for running big code eval :

1. accelerate launch main.py   --model /cb/cold2/ganeshv/studies/efficiency/inferenceTimeCompute/magpieqwencoding_llama3p1_8b_ft/checkpoint_1000_to_hf   --trust_remote_code   --tasks mbpp   --temperature 0.8   --do_sample True   --n_samples 50   --batch_size 18   --precision bf16   --allow_code_execution   --save_generations   --save_generations_path /cb/cold/srinjoym/evals/
2. accelerate launch main.py   --model meta-llama/Llama-3.1-8B-Instruct   --use_auth_token --trust_remote_code   --tasks mbpp   --temperature 0.8   --do_sample True   --n_samples 50   --batch_size 18   --precision bf16   --allow_code_execution   --save_generations   --save_generations_path /cb/cold/srinjoym/llama_evals/
