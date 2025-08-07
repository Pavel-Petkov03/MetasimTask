import os

from fastapi import FastAPI, HTTPException
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from app.schemas import TextRequest, ChatRequest, MemoryOwnerEnum

load_dotenv(verbose=True)
app = FastAPI()
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key= os.environ["API_KEY"],
    max_tokens = 1000,
    temperature = 0.5,
)

@app.post("/clean_text")
async def clean_text(text_request : TextRequest):
    try:
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
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Text cleaning failed: {str(error)}")


@app.post("/chat")
async def chat(chat_request: ChatRequest):
    buyer_prompt = """
    You are a skeptical buyer currently evaluating some product
    - The product will be passed in the first prompt in the format `product:{product}`
    - You must answer in the first prompt in the format `Hi. I am interested in the product: {product}. Tell me why is it good`
    - Be polite but critically examine all claims
    - Ask probing questions about price, quality, and usefulness
    - Never accept immediately without justification
    """
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", buyer_prompt),
            *[("human", message.text) for message in chat_request.memory if message.role == MemoryOwnerEnum.HUMAN],
            *[("assistant", message.text) for message in chat_request.memory if message.role == MemoryOwnerEnum.AI],
            ("human", chat_request.current_message),
        ])

        chain = LLMChain(llm=llm, prompt=prompt)
        props = {
            "product" : chat_request.product
        }
        result = await chain.ainvoke(props)
        return {
            "answer": result["text"]
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(error)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


