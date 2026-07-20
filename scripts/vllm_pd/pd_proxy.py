#!/usr/bin/env python3
"""Minimal OpenAI completions proxy for vLLM disaggregated prefill."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("trie.pd_proxy")


def _prefill_index(
    payload: dict[str, Any], count: int, round_robin: itertools.cycle[int]
) -> int:
    cache_salt = payload.get("cache_salt")
    if isinstance(cache_salt, str) and cache_salt:
        digest = hashlib.sha256(cache_salt.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % count
    return next(round_robin)


def _prefill_payload(payload: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(payload)
    prepared["stream"] = False
    prepared["max_tokens"] = 1
    if "max_completion_tokens" in prepared:
        prepared["max_completion_tokens"] = 1
    prepared.pop("stream_options", None)
    prepared.pop("min_tokens", None)
    prepared.pop("min_completion_tokens", None)
    prepared["return_token_ids"] = False
    prepared["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    return prepared


def _upstream_headers(request: Request) -> dict[str, str]:
    headers = {"X-Request-Id": request.headers.get("x-request-id", str(uuid.uuid4()))}
    authorization = request.headers.get("authorization")
    if authorization:
        headers["Authorization"] = authorization
    return headers


def create_app(prefill_urls: list[str], decode_url: str) -> FastAPI:
    prefill_urls = [url.rstrip("/") for url in prefill_urls]
    decode_url = decode_url.rstrip("/")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=None,
            ),
        )
        app.state.prefill_round_robin = itertools.cycle(range(len(prefill_urls)))
        app.state.prefill_request_counts = [0] * len(prefill_urls)
        app.state.prefill_engine_ids = [None] * len(prefill_urls)
        app.state.decode_request_count = 0
        yield
        await app.state.client.aclose()

    app = FastAPI(title="trie vLLM P/D proxy", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "prefill_count": len(prefill_urls), "decode_count": 1}

    @app.get("/status")
    async def status() -> dict[str, object]:
        return {
            "prefill_urls": prefill_urls,
            "decode_url": decode_url,
            "prefill_request_counts": app.state.prefill_request_counts,
            "prefill_engine_ids": app.state.prefill_engine_ids,
            "decode_request_count": app.state.decode_request_count,
        }

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        response = await app.state.client.get(
            f"{decode_url}/v1/models",
            headers=_upstream_headers(request),
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers={
                "content-type": response.headers.get(
                    "content-type", "application/json"
                )
            },
        )

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request body must be an object")

        prefill_index = _prefill_index(
            payload,
            len(prefill_urls),
            app.state.prefill_round_robin,
        )
        prefill_url = prefill_urls[prefill_index]
        app.state.prefill_request_counts[prefill_index] += 1
        headers = _upstream_headers(request)

        prefill_response = await app.state.client.post(
            f"{prefill_url}/v1/completions",
            json=_prefill_payload(payload),
            headers=headers,
        )
        if not prefill_response.is_success:
            return JSONResponse(
                status_code=prefill_response.status_code,
                content={
                    "error": "prefill request failed",
                    "upstream": prefill_url,
                    "body": prefill_response.text,
                },
            )

        kv_transfer_params = prefill_response.json().get("kv_transfer_params")
        if not kv_transfer_params:
            raise HTTPException(
                status_code=502,
                detail=f"prefill response from {prefill_url} omitted kv_transfer_params",
            )
        remote_engine_id = kv_transfer_params.get("remote_engine_id")
        app.state.prefill_engine_ids[prefill_index] = remote_engine_id
        logger.info(
            "prefill complete index=%d upstream=%s remote_engine_id=%s",
            prefill_index,
            prefill_url,
            remote_engine_id,
        )

        decode_payload = dict(payload)
        decode_payload["kv_transfer_params"] = kv_transfer_params
        decode_request = app.state.client.build_request(
            "POST",
            f"{decode_url}/v1/completions",
            json=decode_payload,
            headers=headers,
        )
        decode_response = await app.state.client.send(decode_request, stream=True)
        app.state.decode_request_count += 1

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for chunk in decode_response.aiter_raw():
                    yield chunk
            finally:
                await decode_response.aclose()

        response_headers = {}
        content_type = decode_response.headers.get("content-type")
        if content_type:
            response_headers["content-type"] = content_type
        return StreamingResponse(
            relay(),
            status_code=decode_response.status_code,
            headers=response_headers,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--prefill", nargs="+", required=True)
    parser.add_argument("--decode", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        create_app(args.prefill, args.decode),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
