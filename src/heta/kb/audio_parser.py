"""Audio and video parsing for Little Heta KB inserts."""

from __future__ import annotations

import base64
import json
from datetime import date
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

from heta.config.schema import HetaConfig
from heta.kb.agent import _chat_completion, _get_chat_model
from heta.providers.clients import build_multimodal_model, resolve_litellm_model_name
from heta.providers.litellm_models import LiteLLMChatModel
from heta.providers.model_protocols import (
    ChatCompletionRequest,
    ChatMessage,
    ChatModelConfig,
    ChatModelOptions,
    ChatModelProtocol,
)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".mp4"}

OPENAI_TRANSCRIBE_MODEL = "gpt-4o-transcribe"

_MIME_TYPES = {
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
    ".mp4": "video/mp4",
}

_QWEN_FORMATS = {
    ".mp3": "mp3",
    ".wav": "wav",
    ".m4a": "m4a",
    ".webm": "webm",
}


def parse_audio_markdown(source_path: Path, archived_path: Path, config: HetaConfig) -> str:
    """Transcribe audio/video and return stable wiki-flavored Markdown."""
    description = transcribe_media(source_path=source_path, config=config)
    media_kind = "Video" if source_path.suffix.lower() == ".mp4" else "Audio"
    return build_audio_markdown(
        title=f"{media_kind} - {source_path.stem}",
        source_name=archived_path.name,
        media_path=f"../../raw/{archived_path.name}",
        media_kind=media_kind,
        summary=description["summary"],
        transcript=description["transcript"],
        key_points_metadata=description["key_points_metadata"],
        interpretation_keywords=description["interpretation_keywords"],
    )


def transcribe_media(*, source_path: Path, config: HetaConfig) -> dict[str, str]:
    if config.llm.provider == "chatgpt":
        transcript = _transcribe_with_openai(source_path, config)
        return _structure_transcript(source_path=source_path, transcript=transcript, config=config)
    if config.llm.provider == "qwen":
        _require_multimodal(config, "Audio/video parsing")
        return _transcribe_with_qwen_omni(source_path, config)
    if config.llm.provider == "custom":
        _require_custom_audio(config)
        return _transcribe_with_custom_audio(source_path, config)
    if config.llm.provider == "gemini":
        _require_multimodal(config, "Audio/video parsing")
        return _transcribe_with_gemini(source_path, config)
    raise ValueError(f"Unsupported audio provider: {config.llm.provider}")


def build_audio_markdown(
    *,
    title: str,
    source_name: str,
    media_path: str,
    media_kind: str,
    summary: str,
    transcript: str,
    key_points_metadata: str,
    interpretation_keywords: str,
) -> str:
    link_label = f"{media_kind} file"
    return (
        "---\n"
        f"title: {title}\n"
        f"sources: [{source_name}]\n"
        f"updated: {date.today().isoformat()}\n"
        "---\n\n"
        "## Summary\n\n"
        f"{summary.strip()}\n\n"
        "## Content\n\n"
        f"[{link_label}](<{media_path}>)\n\n"
        "### Transcript\n\n"
        f"{transcript.strip() or 'No transcript extracted.'}\n\n"
        "### Key Points and Metadata\n\n"
        f"{key_points_metadata.strip()}\n\n"
        "### Interpretation and Keywords\n\n"
        f"{interpretation_keywords.strip()}\n\n"
        "## Related Pages\n\n"
        "- None yet\n\n"
        "## Source\n\n"
        f"- {source_name}\n"
    )


def _transcribe_with_openai(path: Path, config: HetaConfig) -> str:
    client = OpenAI(api_key=config.llm.api_key, timeout=300)
    with path.open("rb") as file:
        result = client.audio.transcriptions.create(
            model=OPENAI_TRANSCRIBE_MODEL,
            file=file,
            response_format="text",
        )
    return str(result).strip()


def _structure_transcript(*, source_path: Path, transcript: str, config: HetaConfig) -> dict[str, str]:
    chat_model = _get_chat_model(config)
    response = _chat_completion(
        chat_model=chat_model,
        messages=[
            {"role": "system", "content": _media_json_system_prompt()},
            {
                "role": "user",
                "content": _structure_user_prompt(filename=source_path.name, transcript=transcript),
            },
        ],
        tools=None,
        temperature=0.1,
        config=config,
    )
    raw = response.message.content or ""
    data = _normalize_description(_extract_json_object(raw))
    if not data["transcript"].strip():
        data["transcript"] = transcript
    return data


def _transcribe_with_qwen_omni(path: Path, config: HetaConfig) -> dict[str, str]:
    return _transcribe_with_openai_compatible_multimodal(path, config, extra_body={"enable_thinking": False})


def _transcribe_with_custom_audio(path: Path, config: HetaConfig) -> dict[str, str]:
    audio_model = _required(config.llm.audio_model, "audio_model")
    chat_model = LiteLLMChatModel(
        ChatModelConfig(
            model_name=resolve_litellm_model_name(
                provider=config.llm.provider,
                model_name=audio_model,
                api_base=config.llm.audio_base_url,
            ),
            api_key=_required(config.llm.audio_api_key, "audio_api_key"),
            api_base=config.llm.audio_base_url,
            request_timeout=300,
        )
    )
    return _transcribe_with_openai_compatible_multimodal(path, config, resolved=chat_model)


def _transcribe_with_openai_compatible_multimodal(
    path: Path,
    config: HetaConfig,
    *,
    extra_body: dict[str, Any] | None = None,
    resolved: ChatModelProtocol | None = None,
) -> dict[str, str]:
    chat_model = resolved or build_multimodal_model(config)
    suffix = path.suffix.lower()
    content: list[dict[str, Any]] = [{"type": "text", "text": _media_prompt(path.name)}]
    if suffix == ".mp4":
        content.append({"type": "video_url", "video_url": {"url": _data_url(path)}})
    else:
        audio_format = _QWEN_FORMATS.get(suffix)
        if audio_format is None:
            raise ValueError(f"Unsupported Qwen audio type: {suffix}")
        content.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": _data_url(path),
                    "format": audio_format,
                },
            }
        )

    response = chat_model.complete(
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=content)],
            options=ChatModelOptions(temperature=0.1, provider_options=extra_body),
        )
    )
    raw = response.message.content or ""
    return _normalize_description(_extract_json_object(raw))


def _transcribe_with_gemini(path: Path, config: HetaConfig) -> dict[str, str]:
    mime = _mime_type(path)
    model = config.llm.multimodal_model
    if not model:
        raise ValueError("Missing LLM multimodal_model in config.")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": _media_prompt(path.name)},
                    {
                        "inline_data": {
                            "mime_type": mime,
                            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {"temperature": 0.1},
    }
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": config.llm.api_key},
        json=payload,
        timeout=300,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Gemini audio transcription failed: HTTP {response.status_code} {response.text[:300]}")
    parts = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    raw = "\n".join(str(part.get("text", "")) for part in parts).strip()
    return _normalize_description(_extract_json_object(raw))


def _media_json_system_prompt() -> str:
    return """You are an audio/video-to-Markdown parser for Little Heta KB inserts.
Return only one valid JSON object. Do not wrap it in Markdown fences.
Keep the transcript faithful. Do not invent details not present in the transcript or media."""


def _media_prompt(filename: str) -> str:
    return f"""Transcribe and describe this audio/video for semantic retrieval.

Filename: {filename}

Return JSON with exactly these string fields:
- summary: one concise paragraph describing the media content.
- transcript: full transcript. Preserve speaker labels and timestamps if available.
- key_points_metadata: important facts, decisions, tasks, names, dates, places, speaker count, duration, language, and media type.
- interpretation_keywords: likely meaning or purpose with uncertainty if needed, ending with compact search keywords.
"""


def _structure_user_prompt(*, filename: str, transcript: str) -> str:
    return f"""Structure this transcript for semantic retrieval.

Filename: {filename}

Transcript:
{transcript}

Return JSON with exactly these string fields:
- summary
- transcript
- key_points_metadata
- interpretation_keywords
"""


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_mime_type(path)};base64,{encoded}"


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = _MIME_TYPES.get(suffix)
    if mime is None:
        raise ValueError(f"Unsupported audio/video type: {suffix}")
    return mime


def _require_multimodal(config: HetaConfig, feature: str) -> None:
    if not (config.llm.multimodal_api_key and config.llm.multimodal_model):
        raise ValueError(
            f"{feature} requires a multimodal model. Run `heta init` and enable custom multimodal API, "
            "or skip this file."
        )


def _require_custom_audio(config: HetaConfig) -> None:
    if not (config.llm.audio_api_key and config.llm.audio_model):
        raise ValueError(
            "Audio/video parsing is not enabled for custom providers because audio APIs vary by vendor. "
            "Use qwen or gemini for built-in audio support, or enable a custom audio adapter later."
        )
    if "/" not in config.llm.audio_model and not config.llm.audio_base_url:
        raise ValueError("Custom audio model names without a LiteLLM provider prefix require audio_base_url.")


def _required(value: str | None, field: str) -> str:
    if not value:
        raise ValueError(f"Missing LLM {field} in config.")
    return value


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Audio model did not return JSON.")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Audio model JSON must be an object.")
    return value


def _normalize_description(data: dict[str, Any]) -> dict[str, str]:
    fields = {
        "summary": "Imported audio or video.",
        "transcript": "No transcript extracted.",
        "key_points_metadata": "No key points extracted.",
        "interpretation_keywords": "Audio or video media; transcript.",
    }
    normalized: dict[str, str] = {}
    for key, fallback in fields.items():
        value = data.get(key)
        normalized[key] = str(value).strip() if value else fallback
    return normalized


__all__ = ["AUDIO_EXTENSIONS", "build_audio_markdown", "parse_audio_markdown", "transcribe_media"]
