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
    Pick the next backend in round-robin order. Each call to this function will
    return a different backend, looping back to the start of the list when we
    reach the end.

    Returns:
        str: The base URL of the next available backend.
    """
    BACKENDS.rotate(-1)        
    return BACKENDS[0]


async def _is_healthy(base_url: str, timeout: float = 1.0) -> bool:
    """
    Check if the backend is healthy by making a request to /api/tags.

    We consider the backend healthy if we can make a successful request to
    /api/tags within the given timeout.

    If aiohttp can't be imported, we assume the backend is healthy.

    Args:
        base_url (str): Base URL of the backend.
        timeout (float, optional): Timeout in seconds. Defaults to 1.0.

    Returns:
        bool: True if the backend is healthy, False otherwise.
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
    Build a ChatOllama client from the given arguments.

    Args:
        backend (str): Base URL of the backend.
        model (str): Model to use.
        temperature (float, optional): Temperature. Defaults to 0.0.
        thinking (Optional[bool], optional): Whether to think before responding. Defaults to None.
        extra_kwargs (Optional[dict], optional): Extra keyword arguments to pass to ChatOllama. Defaults to None.

    Returns:
        ChatOllama: The built ChatOllama client.
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
    Get a ready ChatOllama client that can be used to call the given model.

    If health_check is True, we will check if the backend is healthy by making a
    request to /api/tags before returning a client. If the backend is not healthy,
    we will retry on the next backend.

    If attempts is greater than 1, we will retry the call on the next backend if
    the previous backend fails. We will use an exponential backoff with a little
    jitter to avoid hammering the backends.

    Args:
        model (str, optional): Model to use. Defaults to "gpt-oss:20b".
        temperature (float, optional): Temperature. Defaults to 0.0.
        thinking (bool, optional): Whether to think before responding. Defaults to False.
        attempts (int, optional): Number of attempts to make. Defaults to DEFAULT_ATTEMPTS.
        health_check (bool, optional): Whether to health check the backend. Defaults to True.
        backoff_base (float, optional): Base for the exponential backoff. Defaults to DEFAULT_BACKOFF_BASE.
        backoff_max (float, optional): Maximum backoff time in seconds. Defaults to DEFAULT_BACKOFF_MAX.
        extra_kwargs (Optional[dict], optional): Extra keyword arguments to pass to ChatOllama. Defaults to None.

    Returns:
        ChatOllama: The built ChatOllama client.

    Raises:
        RuntimeError: If all backends failed after the given number of attempts.
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
    Forward a question to the chosen backend.

    Args:
        model (str): Model to use.
        question (Union[str, dict]): Question to ask the backend.
        temperature (float, optional): Temperature. Defaults to 0.0.
        thinking (bool, optional): Whether to think before responding. Defaults to False.
        attempts (int, optional): Number of attempts to make. Defaults to DEFAULT_ATTEMPTS.
        health_check (bool, optional): Whether to check the backend's health. Defaults to True.
        backoff_base (float, optional): Base for the exponential backoff. Defaults to DEFAULT_BACKOFF_BASE.
        backoff_max (float, optional): Maximum backoff time. Defaults to DEFAULT_BACKOFF_MAX.
        timeout (Optional[float], optional): Timeout in seconds. Defaults to DEFAULT_TIMEOUT.
        return_client (bool, optional): Whether to return the client instead of the response. Defaults to False.
        stream (bool, optional): Whether to stream the response. Defaults to False.
        extra_kwargs (Optional[dict], optional): Extra keyword arguments to pass to ChatOllama. Defaults to None.

    Returns:
        Union[ChatOllama, str, AIMessage]: The response from the backend, or the client if return_client is True.
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
        """
        Execute an asynchronous call to the LLM client and process the response.

        If streaming is enabled, the function collects chunks of the response,
        concatenates them, and returns the result either as an AIMessage or a string
        if AIMessage cannot be imported. Otherwise, it returns the result of a direct
        invocation of the LLM client.

        Returns:
            Union[AIMessage, str]: The processed response from the LLM client.
        """

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



