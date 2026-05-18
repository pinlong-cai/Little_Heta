# heta insert

Insert files or directories into the Little Heta knowledge base.

```bash
heta insert ./docs
heta insert report.pdf notes.md
```

What it does:

- Copies original files into `~/.heta/workspace/kb/raw`.
- Parses supported formats into Markdown.
- Runs the wiki merge agent.
- Updates wiki pages under `~/.heta/workspace/kb/wiki`.
- Updates the vector index when vector indexing is enabled.
- Commits wiki changes with Git.

Large PDFs are planned and split by default before parsing. Control that with:

```bash
heta insert-planning status
heta insert-planning on
heta insert-planning off
```

