import torch
from langchain.chains import LLMChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "TheBloke/Llama-2-7b-Chat-GPTQ"

tokenizers = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             torch_dtype=torch.float16)

hf_pipeline = pipeline(
    "text-generation", model=model, tokenizer=tokenizers, max_length=100, temperature=0.7,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

template = "You are a translator. Translate the following English text to French: {text}"
prompt = PromptTemplate(input_variables=["text"], template=template)

chain = LLMChain(llm=llm, prompt=prompt)

output = chain.run("How are you?")
print(output)  # Answer: Comment ça va? (How are you?)
