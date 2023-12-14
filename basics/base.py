import os
import sys

from langchain.llms import OpenAI

# print(sys.argv[1])
os.environ['OPENAPI_API_KEY'] = sys.argv[1]

llm = OpenAI(temperature=0.9, openai_api_key=sys.argv[1], model="text-curie-001")

prompt = "What is the best battle royal games in pc?"

print(llm(prompt))
