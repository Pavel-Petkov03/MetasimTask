from fastapi import FastAPI
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, dotenv_values
from pydantic import BaseModel

load_dotenv(verbose=True)
env_config = dotenv_values(".env")
app = FastAPI()
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key= env_config["API_KEY"],
    max_tokens = 1000,
    temperature = 0.5,
)

class TextRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    text: str
    memory: list[str]


@app.post("/clean_text")
async def clean_text(text_request : TextRequest):
    prompt = ChatPromptTemplate.from_template(
        """
        Remove headers, footers, page numbers from this text.
        Return ONLY the cleaned content:
        
        {input_text}
        """
    )
    chain = prompt | llm
    props = {
        "input_text": text_request.text
    }
    return {"cleared_text": await chain.ainvoke(props)["content"]}

@app.post("/chat")
async def chat(chat_request: ChatRequest):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a skeptical buyer. Be critical for the product but polite at the same time."),
        *[("human", message) for message in chat_request.memory],
        ("human", chat_request.text),
        ("assistant", "{memory}")
    ])
    chain = LLMChain(llm=llm, prompt=prompt)
    props = {
        "memory": chat_request.memory,
    }
    return {"response": await chain.ainvoke(props)["content"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


