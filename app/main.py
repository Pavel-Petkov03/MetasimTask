from fastapi import FastAPI
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, dotenv_values
from app.schemas import TextRequest, ChatRequest

load_dotenv(verbose=True)
env_config = dotenv_values(".env")
app = FastAPI()
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key= env_config["API_KEY"],
    max_tokens = 1000,
    temperature = 0.5,
)

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
    result = await chain.ainvoke(props)
    return {"cleared_text":  result.content}

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
    result = await chain.ainvoke(props)
    return {"response": result["content"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


