from unittest.mock import Mock, patch

from heta.cli.init import _custom_base_url_required
from heta.providers.llm import validate_llm
from heta.providers.mineru import validate_mineru_cloud, validate_mineru_local


@patch("heta.providers.llm.requests.get")
def test_validate_chatgpt_success(get: Mock) -> None:
    get.return_value.status_code = 200

    assert validate_llm("chatgpt", "sk-test") is True


@patch("heta.providers.llm.requests.get")
def test_validate_llm_non_200_fails(get: Mock) -> None:
    get.return_value.status_code = 401

    assert validate_llm("qwen", "bad-key") is False


@patch("heta.providers.llm.requests.get")
def test_validate_custom_llm_uses_base_url_models(get: Mock) -> None:
    get.return_value.status_code = 200

    assert validate_llm("custom", "sk-test", "http://llm.local/v1") is True
    assert get.call_args.args[0] == "http://llm.local/v1/models"


def test_custom_init_requires_base_url_only_for_bare_model_names() -> None:
    assert _custom_base_url_required("local-chat") is True
    assert _custom_base_url_required("anthropic/claude-sonnet-4-5") is False
    assert _custom_base_url_required("openai/text-embedding-3-small") is False


@patch("heta.providers.mineru.requests.post")
def test_validate_mineru_cloud_success(post: Mock) -> None:
    post.return_value.status_code = 200

    assert validate_mineru_cloud("mineru-token") is True


@patch("heta.providers.mineru.requests.get")
def test_validate_mineru_local_success(get: Mock) -> None:
    get.return_value.status_code = 200

    assert validate_mineru_local("http://127.0.0.1:8000") is True
    assert get.call_args.args[0] == "http://127.0.0.1:8000/health"
