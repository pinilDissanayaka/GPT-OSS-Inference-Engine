from collections import deque
from langchain_ollama import ChatOllama
from typing import List
from langchain_core.tools import BaseTool


BACKENDS = deque([
    "http://localhost:11434",
])

async def pick_backend():
    BACKENDS.rotate(0) 
    
    return BACKENDS[0]


async def forward_to_backend(model, question, temperature: int, thinking: bool = False, attempts: int = len(BACKENDS)):
    backend = await pick_backend()
    
    for _ in range(attempts):
        llm = ChatOllama(
            base_url=backend,
            name=model,
            thinking=thinking,
            temperature=temperature
        )
