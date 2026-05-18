# heta vector

Manage document vector indexing.

```bash
heta vector status
heta vector on
heta vector off
```

When enabled, Little Heta syncs wiki chunks into the local sqlite-vec database
after insert. `heta query` can then retrieve relevant page sections before the
agent reads the wiki.

