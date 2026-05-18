# heta clean

Clean generated wiki knowledge while keeping original raw files.

```bash
heta clean
heta clean --yes
```

What it does:

- Clears generated wiki pages.
- Resets `wiki/index.md`.
- Appends a clean operation to `wiki/log.md`.
- Deletes the local wiki vector database.
- Keeps `~/.heta/workspace/kb/raw`.
- Commits the clean operation to the wiki Git repo.

