from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from typing import Iterable, List, Optional, Tuple, Union
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage 


logger = logging.getLogger(__name__)

BACKENDS: deque[str] = deque([
    "http://ollama1:11434",
    "http://ollama2:11434",
])

DEFAULT_ATTEMPTS = len(BACKENDS)
DEFAULT_BACKOFF_BASE = 0.35  
DEFAULT_BACKOFF_MAX = 2.5   
DEFAULT_TIMEOUT = 60        

async def pick_backend() -> str:
    """
    Rotate the deque one step (true round-robin) and return the next backend.
    """
    BACKENDS.rotate(-1)           # move left; new head is the "next" backend
    return BACKENDS[0]


async def _is_healthy(base_url: str, timeout: float = 1.0) -> bool:
    """
    Try to hit Ollama's /api/tags (cheap) to confirm the server responds.
    Uses aiohttp if available; otherwise returns True (skip check).
    """
    try:
        import aiohttp  
    except Exception:
        return True

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/tags", timeout=timeout) as r:
                return r.status == 200
    except Exception:
        return False


def _build_llm(
    *,
    backend: str,
    model: str,
    temperature: float = 0.0,
    thinking: Optional[bool] = None,
    extra_kwargs: Optional[dict] = None,
) -> ChatOllama:
    """
    Centralized constructor so we don't duplicate kwargs. We keep 'name=model'
    (as in your snippet) to match your environment, while also allowing extras.
    """
    kwargs = dict(base_url=backend, model=model, temperature=temperature)
    if thinking is not None:
        kwargs["thinking"] = thinking
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return ChatOllama(**kwargs)


async def get_llm_client(
    model: str="gpt-oss:20b",
    *,
    temperature: float = 0.0,
    thinking: bool = False,
    attempts: int = DEFAULT_ATTEMPTS,
    health_check: bool = True,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    backoff_max: float = DEFAULT_BACKOFF_MAX,
    extra_kwargs: Optional[dict] = None,
) -> ChatOllama:
    """
    Return a connected ChatOllama client from the first healthy / working backend.
    Retries across backends in round-robin order with exponential backoff.
    """
    attempts = max(1, min(attempts, len(BACKENDS)))
    last_err: Optional[BaseException] = None


    order: List[str] = []
    for _ in range(len(BACKENDS)):
        order.append(await pick_backend())

    for i, backend in enumerate(order[:attempts], start=1):
        try:
            if health_check:
                ok = await _is_healthy(backend)
                if not ok:
                    raise ConnectionError(f"Backend not healthy: {backend}")

            llm = _build_llm(
                backend=backend,
                model=model,
                temperature=temperature,
                thinking=thinking,
                extra_kwargs=extra_kwargs,
            )
            if i > 1:
                logger.info("Recovered by switching to backend %s", backend)
            return llm

        except Exception as e:
            last_err = e
            logger.warning("Backend %s failed on attempt %d/%d: %r",
                           backend, i, attempts, e)
            if i < attempts:
                # Exponential backoff with a little jitter
                sleep_s = min(backoff_max, backoff_base * (2 ** (i - 1)))
                sleep_s *= 1 + random.random() * 0.25  # +0–25% jitter
                await asyncio.sleep(sleep_s)

    raise RuntimeError(f"All backends failed after {attempts} attempt(s).") from last_err


async def forward_to_backend(
    model: str,
    question: Union[str, dict], 
    *,
    temperature: float = 0.0,
    thinking: bool = False,
    attempts: int = DEFAULT_ATTEMPTS,
    health_check: bool = True,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    backoff_max: float = DEFAULT_BACKOFF_MAX,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    return_client: bool = False,
    stream: bool = False,
    extra_kwargs: Optional[dict] = None,
) -> Union[ChatOllama, str, AIMessage]:
    """
    High-level helper:
      - If return_client=True, returns a ready ChatOllama client (no call made).
      - Otherwise, calls the model with your `question` and returns the response.
        * If stream=True -> returns an AIMessage streamed to completion.
        * Else -> returns AIMessage or str (depending on lc version), typically use `.content`.

    Notes:
      - We wrap the single call in a timeout (if provided).
      - We rely on `get_llm_client` for retry/failover and only make ONE call.
        (If you want retries on the *call itself*, add a loop here as well.)
    """
    llm = await get_llm_client(
        model=model,
        temperature=temperature,
        thinking=thinking,
        attempts=attempts,
        health_check=health_check,
        backoff_base=backoff_base,
        backoff_max=backoff_max,
        extra_kwargs=extra_kwargs,
    )

    if return_client:
        return llm

    async def _call():
        if stream:

            chunks = []
            async for ev in llm.astream(question):
                if hasattr(ev, "content"):
                    chunks.append(ev.content)
                elif isinstance(ev, str):
                    chunks.append(ev)
            text = "".join(chunks).strip()
            try:
                from langchain_core.messages import AIMessage
                return AIMessage(content=text)
            except Exception:
                return text
        else:
            return await llm.ainvoke(question)

    if timeout and timeout > 0:
        return await asyncio.wait_for(_call(), timeout=timeout)
    return await _call()



