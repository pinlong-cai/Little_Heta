# heta insert-planning

Manage smart insert planning.

```bash
heta insert-planning status
heta insert-planning on
heta insert-planning off
```

When enabled, large PDFs are profiled before parsing. Little Heta samples PDF
metadata, outline, page count, and page text, asks a planning agent for split
ranges, validates the plan, and then parses smaller parts.

