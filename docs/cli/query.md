# heta query

Ask a read-only question against inserted documents.

```bash
heta query "What does the design doc say about rate limits?"
```

Use this when the answer should come from the wiki knowledge base, not personal
memory.

Options:

```bash
heta query "..." --top-k 5
```

`--top-k` controls how many vector matches are offered to the query agent.

