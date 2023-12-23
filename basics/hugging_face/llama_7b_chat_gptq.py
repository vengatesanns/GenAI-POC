from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
model_name_or_path = "TheBloke/Llama-2-13b-Chat-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

print(model)

# tokenizer.model
# tokenizer.json
# special_tokens_map
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

print(tokenizer)

# prompt = "What is the best battle royal games?"
prompt = "What is the best multiplayer games?"
prompt_template = f'''[INST] <<SYS>>
You are a Video Game Shop Owner, suggest best games
<</SYS>>
{prompt}[/INST]
'''

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
print(input_ids)
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(output)
print(tokenizer.decode(output[0]))
