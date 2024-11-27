import os

from langchain.chains import RetrievalQA
from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer

print([e.value for e in Language])

html_text = """
<h1 abc-company>Sample H1 Tag</h1>
<h2 abc-company>Sample H2 Tag</h2>
<h3 abc-company>Sample H3 Tag</h3>
"""

html_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.HTML, chunk_size=60, chunk_overlap=0
)
html_docs = html_splitter.create_documents([html_text])
print(html_docs)

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")
# Initialize HuggingFace embeddings
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# Create and persist a Chroma vector database from the chunked documents
vector_database = Chroma.from_documents(
    documents=html_docs,
    embedding=huggingface_embeddings,
    persist_directory=DB_DIR,
)

vector_database.persist()

# Load the Model
model_name_or_path = "TheBloke/Llama-2-13b-Chat-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
prompt = "Generate H3 tag"
prompt_template = f'''[INST] <<SYS>>
Customized Html Codes Generator
<</SYS>>
{prompt}[/INST]
'''

persist_dir = "./db"
db = Chroma(persist_directory=persist_dir, embedding_function=huggingface_embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)
resp = qa_chain({"query": prompt_template})
print(resp)
