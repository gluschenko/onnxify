# Roadmap Tasks

The repository may contain a `.roadmap` directory with execution-oriented task files.

Use roadmap files when work is large enough to need durable context, staged implementation notes, or handoff between humans and coding agents.

## File Naming

Roadmap task files must use this format:

```text
.roadmap/OXY-XXX.md
```

`XXX` is the task sequence number padded to three digits:

- `1` becomes `001`
- `12` becomes `012`
- `33` becomes `033`
- `123` remains `123`

Examples:

- `.roadmap/OXY-001.md`
- `.roadmap/OXY-012.md`
- `.roadmap/OXY-123.md`

When renaming roadmap files, also update references between roadmap tasks.

## Index File

The `.roadmap` directory must also contain:

```text
.roadmap/INDEX.md
```

`INDEX.md` mirrors the metadata from every roadmap task header in the directory.

It must contain a table with these columns:

- `ID`
- `Title`
- `Status`
- `Estimated complexity`
- `Created`
- `Last modified`

Template:

```markdown
# Roadmap Index

| ID | Title | Status | Estimated complexity | Created | Last modified |
| --- | --- | --- | --- | --- | --- |
| [OXY-001](OXY-001.md) | Task Title | TODO | 8 human-hours | 2026-06-27 | 2026-06-27 |
```

The `ID` column must link to the corresponding task file with a relative Markdown link, for example `[OXY-001](OXY-001.md)`.

Update `INDEX.md` whenever a roadmap task is created, renamed, deleted, or when any mirrored header field changes.

## Required Header

Each roadmap task must start with a level-1 title, followed immediately by a metadata table.

Template:

```markdown
# OXY-001: Task Title

| Field | Value |
| --- | --- |
| Status | TODO |
| Estimated complexity | 8 human-hours |
| Created | 2026-06-27 |
| Last modified | 2026-06-27 |
```

Allowed status values:

- `TODO`
- `IN PROGRESS`
- `REVIEW`
- `CHECKING`
- `DONE`
- `REJECTED`

Use strictly formatted dates in `YYYY-MM-DD` format for `Created`, `Last modified`, and worklog entries. Do not use locale-specific formats such as `6/27/2026`, month names, or relative dates such as `today`.

`Estimated complexity` is an approximate implementation cost in conditional human-hours. It is intentionally rough; use it to keep tasks small enough to execute.

## Body

After the header table, write the task context and implementation plan in ordinary Markdown.

Recommended sections:

- `Summary`
- `Depends On`
- `Enables`
- `Goals`
- `Non-Goals`
- `Implementation Steps`
- `Tests`
- `Acceptance Criteria`
- `Risks`

Use only the sections that help the task. Keep roadmap tasks execution-sized rather than epic-sized.

## Worklog

Every roadmap file must end with a `## Worklog` section.

The worklog records chronological notes, actions, clarifications, decisions, and implementation nuances discovered while executing the task.

For human executors, worklog entries are optional but encouraged.

For coding agents such as Codex, Claude Code, or other software agents, worklog entries are mandatory. The point is agent handoff and durable context, not attachment to any particular model or vendor.

Recommended entry format:

```markdown
## Worklog

- 2026-06-27: Created the task and captured initial scope.
- 2026-06-28: Implemented metadata extraction; noted that private field reflection should remain opt-in.
```

Keep entries concise, factual, and useful to the next executor.

## Maintenance Rules

- Update `Last modified` whenever the task text changes materially.
- Update `.roadmap/INDEX.md` whenever mirrored task metadata changes.
- Keep `Created` stable after the file is created.
- Move `Status` forward as work progresses.
- Set `REJECTED` only when the task is intentionally abandoned or superseded.
- Set `DONE` only when acceptance criteria are met.
- Prefer splitting a large roadmap task into multiple numbered files over expanding it indefinitely.
