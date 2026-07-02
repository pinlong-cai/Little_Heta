import sys
from pathlib import Path
from types import SimpleNamespace

from heta.config.schema import HetaConfig, InsertPlanningConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.kb.audio_parser import build_audio_markdown, transcribe_media
from heta.kb.parser import parse_document
from heta.kb.text import extract_title


def _config() -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(provider="qwen", api_key="sk-test"),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=False),
        insert_planning=InsertPlanningConfig.enabled(),
    )


def _chatgpt_config() -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(provider="chatgpt", api_key="sk-test"),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=False),
        insert_planning=InsertPlanningConfig.enabled(),
    )


def _custom_without_multimodal_config() -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(
            provider="custom",
            api_key="sk-test",
            chat_api_key="sk-chat",
            chat_model="chat-model",
            chat_base_url="http://chat.local/v1",
            embedding_api_key="sk-embedding",
            embedding_model="embedding-model",
            embedding_base_url="http://embedding.local/v1",
        ),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=False),
        insert_planning=InsertPlanningConfig.enabled(),
    )


def _custom_with_audio_config() -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(
            provider="custom",
            api_key="sk-test",
            chat_api_key="sk-chat",
            chat_model="chat-model",
            chat_base_url="http://chat.local/v1",
            embedding_api_key="sk-embedding",
            embedding_model="embedding-model",
            embedding_base_url="http://embedding.local/v1",
            audio_api_key="sk-audio",
            audio_model="audio-model",
            audio_base_url="http://audio.local/v1",
        ),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=False),
        insert_planning=InsertPlanningConfig.enabled(),
    )


def test_build_audio_markdown_uses_compact_retrieval_sections() -> None:
    markdown = build_audio_markdown(
        title="Audio - Meeting",
        source_name="meeting.mp3",
        media_path="../../raw/meeting.mp3",
        media_kind="Audio",
        summary="A meeting recording.",
        transcript="Speaker 1: Let's ship the feature.",
        key_points_metadata="Decision: ship the feature. Language: English.",
        interpretation_keywords="Meeting notes. keywords: feature, release.",
    )

    assert extract_title(markdown, "fallback") == "Audio - Meeting"
    assert "[Audio file](<../../raw/meeting.mp3>)" in markdown
    assert "### Transcript" in markdown
    assert "### Key Points and Metadata" in markdown
    assert "### Interpretation and Keywords" in markdown
    assert "## Related Pages" in markdown
    assert "## Source" in markdown


def test_chatgpt_audio_transcribes_then_structures_transcript(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"mp3 bytes")
    seen: dict[str, object] = {}

    class FakeTranscriptions:
        @staticmethod
        def create(**kwargs):
            seen["transcription"] = kwargs
            return "Speaker: hello."

    class FakeOpenAIClient:
        audio = SimpleNamespace(transcriptions=FakeTranscriptions())

    class FakeOpenAIFactory:
        def __init__(self, **kwargs):
            seen["openai_kwargs"] = kwargs

        audio = FakeOpenAIClient.audio

    chat_model = object()
    monkeypatch.setattr("heta.kb.audio_parser.OpenAI", FakeOpenAIFactory)
    monkeypatch.setattr("heta.kb.audio_parser._get_chat_model", lambda config: chat_model)

    def fake_chat_completion(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(
                content=(
                    '{"summary":"A meeting.","transcript":"Speaker: hello.",'
                    '"key_points_metadata":"Language: English.",'
                    '"interpretation_keywords":"meeting, test"}'
                )
            )
        )

    monkeypatch.setattr("heta.kb.audio_parser._chat_completion", fake_chat_completion)

    result = transcribe_media(source_path=audio, config=_chatgpt_config())

    assert result["summary"] == "A meeting."
    assert seen["openai_kwargs"]["api_key"] == "sk-test"
    assert seen["transcription"]["model"] == "gpt-4o-transcribe"
    assert seen["transcription"]["response_format"] == "text"
    assert seen["chat_model"] is chat_model
    assert "Speaker: hello." in seen["messages"][1]["content"]


def test_build_audio_markdown_supports_video_link_label() -> None:
    markdown = build_audio_markdown(
        title="Video - Demo",
        source_name="demo.mp4",
        media_path="../../raw/demo.mp4",
        media_kind="Video",
        summary="A product demo.",
        transcript="Narrator: This is the dashboard.",
        key_points_metadata="Media type: video.",
        interpretation_keywords="Product demo, dashboard.",
    )

    assert "[Video file](<../../raw/demo.mp4>)" in markdown


def test_parse_document_accepts_audio_branch(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp3"
    archived = tmp_path / "raw_meeting.mp3"
    source.write_bytes(b"mp3")
    archived.write_bytes(b"mp3")

    monkeypatch.setattr(
        "heta.kb.parser.parse_audio_markdown",
        lambda source_path, archived_path, config: build_audio_markdown(
            title="Audio - Meeting",
            source_name=archived_path.name,
            media_path="../../raw/raw_meeting.mp3",
            media_kind="Audio",
            summary="A meeting.",
            transcript="Speaker 1: hello.",
            key_points_metadata="Language: English.",
            interpretation_keywords="meeting, test",
        ),
    )

    document = parse_document(source, archived, _config())

    assert document.title == "Audio - Meeting"
    assert document.source_name == "raw_meeting.mp3"
    assert document.metadata["extension"] == ".mp3"
    assert "### Transcript" in document.markdown_content


def test_audio_is_disabled_for_custom_without_audio_adapter(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"mp3")

    try:
        parse_document(source, source, _custom_without_multimodal_config())
    except ValueError as exc:
        assert "Audio/video parsing is not enabled for custom providers" in str(exc)
        assert "audio APIs vary by vendor" in str(exc)
    else:
        raise AssertionError("audio parsing should require multimodal config")


def test_custom_audio_uses_audio_adapter(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"mp3 bytes")
    seen: dict[str, object] = {}

    class FakeLiteLLM:
        @staticmethod
        def completion(**kwargs):
            seen["request"] = kwargs
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"summary":"A meeting.","transcript":"Speaker: hello.",'
                                '"key_points_metadata":"Language: English.",'
                                '"interpretation_keywords":"meeting, test"}'
                            ),
                        }
                    }
                ]
            }

    monkeypatch.setitem(sys.modules, "litellm", FakeLiteLLM)

    result = transcribe_media(source_path=audio, config=_custom_with_audio_config())

    assert result["summary"] == "A meeting."
    assert seen["request"]["api_key"] == "sk-audio"
    assert seen["request"]["api_base"] == "http://audio.local/v1"
    assert seen["request"]["model"] == "openai/audio-model"
    content = seen["request"]["messages"][0]["content"]
    assert content[1]["type"] == "input_audio"
