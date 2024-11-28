import torch
from langchain.chains import LLMChain
from langchain_community.document_loaders import CSVLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.cuda.empty_cache()
torch.cuda.memory_summary(device="cuda")

file_path = "C:\\Users\\venga\\Downloads\\archive (1)\\test-data.csv"
# df = pd.read_csv(file_path, header="infer")
# print(df.head())

loader = CSVLoader(file_path=file_path)
documents = loader.load()
print(documents[:5])

# model_name = "TheBloke/Llama-2-7b-Chat-GPTQ"
model_name = "openai-community/gpt2-medium"

tokenizers = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             torch_dtype=torch.float16)

hf_pipeline = pipeline(
    "text-generation", model=model, tokenizer=tokenizers, max_length=50, temperature=0.7,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

prompt_template = """
You are a car expert. Based on the following car dataset, answer the question:

Dataset: {data}

Question: list car details which have less then 30k kilometers driven.
"""

question_1_prompt = PromptTemplate(
    input_variables=["data"],
    template=prompt_template,
)

chain_1 = LLMChain(llm=llm, prompt=question_1_prompt)

csv_data_as_string = "\n".join([str(doc) for doc in documents])
response_1 = chain_1.run(data=csv_data_as_string)
print("\n--- AI Response for Question 1 ---")
print(response_1)
