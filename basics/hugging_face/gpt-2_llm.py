from langchain.chains import LLMChain
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.prompts import PromptTemplate

gpt2_llm = HuggingFaceHub(
    repo_id="gpt2",
    model_kwargs={'temperature': 0.8, 'max_length': 100}
)

prompt = PromptTemplate(
    input_variables=["profession"],
    template="you had a job! you are the {profession} and you didn't have to be sarcastic"
)

llm_chain = LLMChain(prompt=prompt, llm=gpt2_llm, verbose=True)
print(llm_chain.run("customer service agent"))
print(llm_chain.run("politician"))
