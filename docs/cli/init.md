# heta init

Set up Little Heta for the first time.

```bash
heta init
```

What it does:

- Creates `~/.heta/heta.yaml`.
- Configures one LLM provider: Qwen, ChatGPT, Gemini, or custom.
- Optionally configures MinerU for PDF and Office parsing.
- Enables vector indexing and insert planning by default.
- Installs the Little Heta skill into Codex and Claude Code.

Prepare before running:

- Your LLM provider API key.
- For custom providers, prepare chat and embedding API settings. Model names
  with a provider prefix such as `openai/gpt-5.4-nano` are treated as
  LiteLLM-native and do not require a base URL. Bare model names require an
  OpenAI-compatible `/v1` base URL.
- Optional custom multimodal settings if you want image parsing.
- Optional MinerU API key from https://mineru.net/apiManage/docs.
