from fastapi import FastAPI
import uvicorn
from utils import get_llm_client


app= FastAPI()


@app.post("/ask")
async def chat(question:str):
    """
    Chat with the AI model.

    Args:
        question (str): The question to ask the AI model.

    Returns:
        str: The response from the AI model.
    """
    llm_client = await get_llm_client()

    response = await llm_client.invoke(question)
    
    print(response)

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
