from heta.config.schema import InsertPlanningConfig, HetaConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.mem.recall import LayerEvidence
from heta.providers.model_protocols import ChatCompletionResult, ChatMessage
import heta.query.smart_query as smart_query_module
from heta.query.smart_query import _parse_text_tool_calls, smart_query


def _config() -> HetaConfig:
    return HetaConfig(
        version=1,
        llm=LLMConfig(provider="qwen", api_key="sk-test"),
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig(enable=False),
        insert_planning=InsertPlanningConfig.enabled(),
    )


def _response(content: str):
    return ChatCompletionResult(
        message=ChatMessage(role="assistant", content=content),
        model_name="fake-model",
    )


def test_parse_text_tool_call_without_parameter_closing_tag() -> None:
    calls = _parse_text_tool_calls(
        "用户问的是...\n"
        "<tool_call> <function=recall_memory> <parameter=query> 工业智能中枢是什么</tool_call>"
    )

    assert len(calls) == 1
    assert calls[0].function.name == "recall_memory"
    assert calls[0].function.arguments == '{"query": "工业智能中枢是什么"}'


def test_smart_query_executes_text_tool_call_instead_of_returning_it(monkeypatch) -> None:
    responses = iter(
        [
            _response(
                "用户问的是...\n"
                "<tool_call> <function=recall_memory> <parameter=query> 工业智能中枢是什么</tool_call>"
            ),
            _response("工业智能中枢是一个用于工业场景的数据与智能能力汇聚平台。"),
        ]
    )
    seen_messages = []

    def fake_chat(chat_model, messages, *, tools):
        seen_messages.append(messages)
        return next(responses)

    def fake_build_chat_model(config):
        return object()

    def fake_retrieve_evidence(query, config, top_k):
        assert query == "工业智能中枢是什么"
        return [LayerEvidence(layer="raw", items=[{"score": 0.9, "text_content": "工业智能中枢是工业数据与智能能力平台。"}])]

    monkeypatch.setattr(smart_query_module, "_chat", fake_chat)
    monkeypatch.setattr(smart_query_module, "build_chat_model", fake_build_chat_model)
    monkeypatch.setattr(smart_query_module, "retrieve_evidence", fake_retrieve_evidence)

    result = smart_query("工业智能中枢是什么", _config())

    assert result.answer == "工业智能中枢是一个用于工业场景的数据与智能能力汇聚平台。"
    assert "<tool_call>" not in result.answer
    assert result.agent_steps == ["recall_memory"]
    assert seen_messages[1][-1]["role"] == "tool"
    assert "工业数据与智能能力平台" in seen_messages[1][-1]["content"]
