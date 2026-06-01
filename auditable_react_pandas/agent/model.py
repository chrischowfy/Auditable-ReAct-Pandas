from __future__ import annotations

import os
import time
from typing import Any

import tiktoken
from openai import AuthenticationError, BadRequestError, NotFoundError, OpenAI, PermissionDeniedError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential


def _is_retryable_openai_error(exc: BaseException) -> bool:
    non_retryable = (AuthenticationError, BadRequestError, PermissionDeniedError, NotFoundError)
    return not isinstance(exc, non_retryable)


def _max_retry_attempts() -> int:
    try:
        return max(1, int(os.environ.get("LLM_MAX_RETRIES", "3")))
    except Exception:
        return 3


class Model:
    """Small OpenAI-compatible chat client used by the ReAct-Pandas agent."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.provider = self.get_provider(model_name)
        self.context_limit = self.get_context_limit(model_name)
        request_timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT", "90"))

        if self.provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPSEEK_API_KEY must be set for DeepSeek models.")
            self.client = OpenAI(
                base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                api_key=api_key,
                timeout=request_timeout,
            )
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            kwargs: dict[str, Any] = {"timeout": request_timeout}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            self.client = OpenAI(**kwargs)

        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def get_provider(model_name: str) -> str:
        name = str(model_name).lower()
        if "deepseek" in name:
            return "deepseek"
        return "openai"

    @staticmethod
    def get_context_limit(model_name: str) -> int:
        name = str(model_name).lower()
        if "deepseek" in name:
            return 64000
        if "claude" in name or "anthropic" in name:
            return 200000
        if "gpt-5" in name or "gpt-4" in name or "gpt-4o" in name:
            return 128000
        if "gpt-3.5" in name:
            return 16385
        return 128000

    def query(self, prompt: str, **kwargs: Any) -> str:
        if not prompt:
            return "Contents must not be empty."
        return self.query_openai(prompt, **kwargs)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(_max_retry_attempts()),
        retry=retry_if_exception(_is_retryable_openai_error),
    )
    def query_openai_with_retry(self, messages: list[dict[str, str]], **kwargs: Any):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs,
        )

    def query_openai(
        self,
        prompt: str,
        system: str | None = None,
        rate_limit_per_minute: int | None = None,
        **kwargs: Any,
    ) -> str:
        if system is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

        response = self.query_openai_with_retry(messages, **kwargs)
        message = response.choices[0].message
        content = message.content or ""

        if self.provider == "deepseek" and not content.strip():
            content = self._repair_empty_deepseek_content(prompt, message, **kwargs)

        if rate_limit_per_minute:
            time.sleep(60 / rate_limit_per_minute)
        return content

    def _repair_empty_deepseek_content(self, prompt: str, message: Any, **kwargs: Any) -> str:
        reasoning = getattr(message, "reasoning_content", None) or ""
        lower_prompt = prompt.lower()
        is_json_action_prompt = (
            "return valid json only" in lower_prompt
            and '"action"' in lower_prompt
            and "pandas_code" in lower_prompt
        )
        is_code_prompt = any(token in lower_prompt for token in [
            "print(ans)",
            "available dataframes",
            "output only the code block",
            "output exactly one markdown code block",
            "pandas expert",
        ])

        if is_json_action_prompt:
            repair_prompt = (
                "Your previous answer had empty final content. "
                "Return valid JSON only, with no markdown and no explanation.\n"
                "Use this schema exactly:\n"
                '{"thought": "one short reason", "action": "pandas_code", '
                '"code": "python code assigning step_result"}\n\n'
                f"Original task:\n{prompt}\n\n"
            )
            if reasoning.strip():
                repair_prompt += f"Prior reasoning (may help, do not repeat it):\n{reasoning}\n\n"
            repair_response = self.query_openai_with_retry(
                [{"role": "user", "content": repair_prompt}],
                **kwargs,
            )
            return (repair_response.choices[0].message.content or "").strip()

        if is_code_prompt:
            repair_prompt = (
                "Your previous answer had empty final content. "
                "Return EXACTLY ONE executable ```python``` code block and nothing else.\n\n"
                f"Original task:\n{prompt}\n\n"
            )
            if reasoning.strip():
                repair_prompt += f"Prior reasoning (may help):\n{reasoning}\n\n"
            repair_prompt += "Remember: assign the final result to `ans` and end with `print(ans)`."
            repair_response = self.query_openai_with_retry(
                [{"role": "user", "content": repair_prompt}],
                **kwargs,
            )
            return (repair_response.choices[0].message.content or "").strip()

        return reasoning

    def get_token_count(self, prompt: str) -> int:
        if not prompt:
            return 0
        return len(self.tokenizer.encode(prompt))
