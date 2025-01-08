from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import lade 
lade.augment_all()
#For a 7B model, set LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7 
lade.config_lade(LEVEL=4, WINDOW_SIZE=7, GUESS_SET_SIZE=5, DEBUG=1, POOL_FROM_PROMPT=True)

assert torch.cuda.is_available()

torch_device = "cuda"

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=torch_device)
model.tokenizer = tokenizer
prompt = "How do you fine tune a large language model?"
input_text = (
    f"<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"
)


model_inputs = tokenizer(input_text, return_tensors='pt').to(torch_device)

#warm up
greedy_output = model.generate(**model_inputs, max_new_tokens=16)
#end warm up

print(f'{greedy_output=}')
print('Done.')
