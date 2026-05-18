# heta ask

Ask using memory and inserted documents together.

```bash
heta ask "How does our auth flow refresh tokens?"
```

This is the default command for agent workflows. It can:

- Search saved memory first.
- Query the document wiki when memory is not enough.
- Store distilled KB insights for later reuse.

Options:

```bash
heta ask "..." --top-k 5 --debug
```

`--debug` shows agent steps, memory evidence, and KB output.

