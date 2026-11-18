"""Sprints LiteLLM client.

https://management.sprints.ai/litellm/ serves the LiteLLM proxy Swagger UI, and the
LiteLLM proxy exposes OpenAI-compatible routes (POST /chat/completions, GET /models,
`Authorization: Bearer <key>`). We use plain `requests`, no OpenAI SDK needed.
The model ID is never guessed: it must come from LITELLM_MODEL.
"""
from __future__ import annotations

from typing import Any

import requests


class LLMError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class LLMConfigError(LLMError):
    pass


class LiteLLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, *, timeout: int = 300,
                 json_mode: bool = False, max_tokens: int | None = None, session: Any | None = None):
        self.base_url = (base_url or "").strip().rstrip("/")
        self._api_key = (api_key or "").strip()
        self.model = (model or "").strip()
        self.timeout = timeout
        self.json_mode = json_mode
        self.max_tokens = max_tokens
        self._session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        if not self.base_url:
            raise LLMConfigError("LITELLM_BASE_URL is not set.")
        if not self._api_key:
            raise LLMConfigError("LITELLM_API_KEY is not set.")
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def list_models(self) -> list[str]:
        """GET /models - lets you discover the exact model ID to put in .env."""
        resp = self._request("get", f"{self.base_url}/models")
        data = resp.get("data", resp) if isinstance(resp, dict) else resp
        return sorted(str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id"))

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        if not self.model:
            raise LLMConfigError("LITELLM_MODEL is empty. Set the exact model ID in .env "
                                 "(use 'List available models' in the app to see valid IDs).")
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        body = self._request("post", f"{self.base_url}/chat/completions", json=payload)
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected response shape from LiteLLM (no choices[0].message.content).") from exc
        if isinstance(content, list):  # some providers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not content or not str(content).strip():
            raise LLMError("The model returned an empty response.")
        if choice.get("finish_reason") == "length":
            raise LLMError("The model output was cut off (finish_reason=length). Use a model with a "
                           "larger output limit or set LITELLM_MAX_TOKENS.")
        return str(content)

    # ------------------------------------------------------------------
    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        headers = self._headers()
        try:
            resp = getattr(self._session, method)(url, headers=headers, timeout=self.timeout, **kwargs)
        except requests.Timeout as exc:
            raise LLMError("The LiteLLM request timed out.") from exc
        except requests.ConnectionError as exc:
            raise LLMError("Could not connect to LiteLLM. Check LITELLM_BASE_URL and your connection.") from exc
        except requests.RequestException as exc:
            raise LLMError(f"Request failed: {type(exc).__name__}") from exc
        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError as exc:
                raise LLMError("LiteLLM returned a non-JSON response.", 200) from exc
        detail = self._detail(resp)
        hints = {
            400: "Bad request. If the message mentions response_format, set LITELLM_JSON_MODE=false.",
            401: "Invalid API key (401).",
            403: "Key not allowed to use this model/route (403).",
            404: "Route or model not found (404). Check LITELLM_BASE_URL and LITELLM_MODEL.",
            429: "Rate limit or budget reached (429).",
        }
        hint = hints.get(resp.status_code, f"LiteLLM returned HTTP {resp.status_code}.")
        raise LLMError(f"{hint} {detail}".strip(), resp.status_code)

    @staticmethod
    def _detail(resp: Any) -> str:
        try:
            body = resp.json()
            err = body.get("error", body) if isinstance(body, dict) else body
            return (str(err.get("message", err)) if isinstance(err, dict) else str(err))[:400]
        except ValueError:
            return (getattr(resp, "text", "") or "")[:300]
