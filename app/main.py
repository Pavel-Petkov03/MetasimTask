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
            You are an advanced text cleaning assistant. Your task is to process text segments by:
            1. Removing all headers, footers, and page numbers (like "/ 8" or "Page 3")
            2. Eliminating any document structure artifacts (section markers, bullet points that don't belong to content)
            3. Preserving all meaningful content exactly as written
            4. Maintaining original formatting of the actual content (paragraphs, lists that are part of content)
            5. Joining broken sentences that were split by page breaks or formatting
            6. Removing any line breaks that interrupt the natural flow of text
            7. Keeping all technical terms, names, and specialized vocabulary unchanged
            
            Now process this text segment:
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


