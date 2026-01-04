from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from prompts import context, code_parser_template
from code_reader import code_reader
from dotenv import load_dotenv
import os
import ast

load_dotenv()

llm = Ollama(
    model="mistral",
    request_timeout=30
)

# result = llm.complete("Hello world")
# print(result)

parser = LlamaParse(result_type="markdown")

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

embed_model = resolve_embed_model("local:BAAI/bge-m3")

vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

# resutlt = query_engine.query("what are some of the routes in the api?")
# print(resutlt)

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="This gives documentation about code for an API, Use this for reading docs"
        ),
    ),
    code_reader,
]


agent = ReActAgent.from_tools(
    tools, llm=llm, verbose=True, context=""
)

code_llm = Ollama(model="codellama")
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)


class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
json_prompt_template = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_template, llm])

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            result = agent.querey(prompt)
            # print(result)
            next_result = output_pipeline.run(response=result)
            # print(next_result)
            cleaned_json = ast.literal_eval(str(next_result).replace("assitant:", ""))
        except Exception as e:
            retries += 1
            print(f"Error occured, retry #{retries}:", e)
        
    if retries >= 3:
        print("Unable to process the rquest, try again ....")
        continue

    print('code generated')
    print(cleaned_json["code"])

    print("\n\nDescription:", cleaned_json["description"])

    filenename = cleaned_json["filename"]

    try:
        with open(os.path.join("output", filenename), "w") as f:
            f.write(cleaned_json["code"])
        print("Saved file", filenename)
    except Exception as e:
        print("Error saving file....")


