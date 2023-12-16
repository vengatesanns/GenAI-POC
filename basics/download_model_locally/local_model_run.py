from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

# model_id = 'google/flan-t5-small'
# model_id = 'google/flan-t5-xxl'
model_id = 'google/flan-t5-large'
# model_id = 'microsoft/DialoGPT-large'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=128
)

local_llm = HuggingFacePipeline(pipeline=pipeline)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt,
                     llm=local_llm
                     )

# question = "What is the capital of India?"
question = "Name some battle royal games list?"

print(llm_chain.run(question))
