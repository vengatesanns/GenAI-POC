from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

# Load the Hugging Face Repo Model
hub_llm = HuggingFaceHub(repo_id="mrm8488/t5-base-finetuned-wikiSQL")

# Create the Prompt Template
prompt = PromptTemplate(
    input_variables=["user_question"],
    template="translate English to SQL: {user_question}"
)

# Create a LLM Chain
llm_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)

# Run the LLM Chain to get the response
print(llm_chain.run("How many models were finetuned using BERT as base model?"))
print(llm_chain.run("How to find the top second max salary in the organization?"))

# Very Lame and LOL answers, Hahahahahahah
# > Entering new LLMChain chain...
# Prompt after formatting:
# translate English to SQL: How to find the top second max salary in the organization?
#
# > Finished chain.
# SELECT MAX Second max salary FROM table
