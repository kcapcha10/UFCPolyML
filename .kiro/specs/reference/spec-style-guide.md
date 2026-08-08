# Spec Style + Structure Guide

Extracted from the RBA M1 LLD reference documents. Writers MUST match this format, density, and level of detail.

---

## 1. requirements.md — Format and Conventions

### Section Hierarchy

```
# Requirements Document       (H1 — document title)
## Introduction                (H2 — free prose, 1-2 paragraphs summarizing scope)
## Glossary                    (H2 — bold term + colon + definition, one per bullet)
## Requirements                (H2 — container, then Bucket headers as bold + H3 requirements)
```

Requirements are grouped into **Buckets** introduced by bold text and a horizontal rule (`---`):

```
---

**Bucket A — <Name>** (Requirements N–M)

### Requirement N: <Short Title>
```

### Requirement Structure

Each requirement contains, in order:

1. **User Story** — bold label, one sentence in standard format: "As a [role], I want [goal], so that [benefit]."
2. **Requirement** — bold label, 1-3 sentences of imperative specification prose. Uses SHALL for mandates.
3. **`#### Acceptance Criteria`** — H4 heading, followed by a numbered list.

### Acceptance Criteria Format

Uses **EARS-style** (Event-Action-Response-State) SHALL statements. Pattern:

```
N. WHEN/IF/WHERE/WHILE <condition>, THE <Subject> SHALL <action>.
```

Conditional keywords: `WHEN`, `IF...THEN`, `WHERE`, `WHILE`. These are UPPERCASED. The subject (actor) is a glossary term (underscore-joined, e.g., `THE Ingestion_Adapter`). The verb is `SHALL` (mandatory), `SHALL NOT` (prohibition), or `MAY` (optional). Multi-clause criteria use commas.

Edge cases and error conditions are expressed as additional numbered acceptance criteria within the same list, using `IF` conditions:

```
5. IF the RMS source returns an access-denied or throttling response, THEN THE Ingestion_Adapter
   SHALL report the error to the caller.
```

### Numbering and IDs

- Requirements are numbered sequentially: `Requirement 1`, `Requirement 2`, etc.
- Acceptance criteria are numbered per requirement: `1.`, `2.`, etc.
- Cross-references use `Requirements N.M` format (e.g., "Requirements 1.2" = Req 1, criterion 2).
- Retired requirements note the removal with a blockquote and retain the number.

### Quantitative Density

- 11 requirements across 5 buckets
- ~4,525 words total
- Avg 8.4 acceptance criteria per requirement (range: 5–17)
- Each criterion is 1-3 lines
- Introduction: ~150 words
- Glossary: ~15 terms

### Standing Decisions Section

A final section outside the requirements proper, listing architecture decisions with named fallback triggers:

```
## Standing Decisions with Named Fallbacks

1. **<Decision title>.** <Current state>. If <specific measurable signal>, evaluate <named fallback> before changing anything else.
```

---

## 2. design.md — Format and Conventions

### Section Hierarchy

```
# Design Document              (H1 — document title)
## Overview                    (H2 — 3-5 paragraphs: scope, deployment target, two "load-bearing ideas")
## Architecture                (H2 — mermaid diagram + explanatory prose)
### Knowledge base topology    (H3 — bullet list of KBs with chunking specs)
### Deployment model and authentication (H3)
### Development-to-production deltas (H3 — table)
### Test ladder                (H3 — table)
## Components and Interfaces   (H2)
### Component N — <Name> (<LLM usage>) (H3 — e.g., "Component 1 — Ingestion Adapter (no LLM)")
### Stage X — <Name> (Task N)  (H3 — for pipeline stages)
## Data Models                 (H2 — code-fenced pseudo-schemas)
## Error Handling              (H2 — ranked by danger, prose)
## Correctness Properties      (H2 — numbered properties)
## Testing Strategy            (H2 — bullet categories)
## Standing Decisions with Named Fallbacks (H2)
```

### Architecture Diagram

Uses **mermaid `flowchart TD`** with:
- Named subgraphs per pipeline stage (labeled with task number)
- `direction LR` inside subgraphs for horizontal internal flow
- Italic annotations in `<br/><i>text</i>` for conditional notes
- Dashed arrows (`-.->`) for feedback/retry flows
- Named intermediate nodes for data products between stages

### Component Specification Style

Each component section includes:
- Purpose statement (1-2 sentences)
- What it does NOT do (negative constraints, stated explicitly)
- Interface/contract description in prose
- **Design consequences** sub-section listing retired code/modules when applicable

Components are labeled with their LLM usage: `(no LLM)` or the model name.

### Data Models

Specified as **pseudo-type code blocks** (not language-tagged):

```
TypeName {
  field_name:  type                        // comment
  field_name:  ENUM_A | ENUM_B | ENUM_C
  field_name:  OtherType[]                 // collection
}
```

Right-aligned comments. No constructor or method signatures. Fields only.

### How Design Maps to Requirements

- Correctness Properties section: each property states `**Validates: Requirements N.M**`
- Component headings reference task numbers: `(Task N)`
- Inline prose references: `(Requirements N.M, N.M)`

### Level of Implementation Specificity

**Very high.** Names exact:
- Python file paths: `retrieval/selfcrag/policy.py`
- Function names: `find_serial_candidates`, `find_manual_tasks`
- Class/type names: `TaskGraph`, `DeviationRecord`, `BucketAssignment`
- Configuration tables with starting defaults
- Specific model names and versions: "Haiku 4.5", "Opus 5"

### Tables

Used for: test ladder, configuration defaults, axis-to-analyzer bindings, dev-to-prod deltas.

```
| Column | Column | Column |
|---|---|---|
| value | value | value |
```

### Quantitative Density

- ~6,958 words total
- 7 major components/stages described
- 21 correctness properties
- 8 data model schemas
- 1 mermaid architecture diagram
- Multiple configuration tables

---

## 3. tasks.md — Format and Conventions

### Section Hierarchy

```
# Implementation Plan: <Project> — <Milestone> (<Scope>)    (H1)
## Overview                    (H2 — 2-3 paragraphs restating architecture + task structure)
## Tasks                       (H2 — the task list)
## Module Ownership Table      (H2 — table)
## Notes                       (H2 — bullet list of standing rules)
## LOE Summary                 (H2 — one paragraph)
## Task Dependency Graph       (H2 — ASCII text block)
```

### Task Numbering and Nesting

Two-level numbering with checkboxes:

```
- [x] N. <Phase Title> (LOE: X days)
  **Phase LOE: X days**
  - [x] N.M <Task title>
    - <Implementation detail bullet>
    - _Requirements: N.M, N.M_
```

- Top-level: `- [x] N.` or `- [ ] N.` — a phase/milestone
- Sub-tasks: `- [x] N.M` — individual work items
- `[x]` = complete, `[ ]` = incomplete
- Phase LOE stated twice: in the top-level line AND as a bold standalone line

### Task Content

Each sub-task contains:
1. A title line (imperative verb: "Implement...", "Write...", "Build...", "Wire...")
2. 1-3 implementation detail bullets (indented under the sub-task)
3. A requirements traceability line in italics: `_Requirements: 1.1, 2.4, 2.5_`

### Task Granularity

- One sub-task = one focused implementation concern (a function, a test, a config change)
- ~2-5 sub-tasks per phase
- Implementation bullets are specific: name files, data models, behaviors
- Property test tasks repeat the property text verbatim and name the tag convention

### Verification Sub-Steps

Property tests and unit tests are **separate sub-tasks**, not embedded in implementation tasks:

```
- [x] 2.4 Write property test for ingestion provenance
  - **Property 1: Provenance completeness** — <property text verbatim>
  - Tag: `Feature: rba-m1-lld, Property 1: Provenance completeness`; minimum 100 iterations
  - **Validates: Requirements 1.2**
```

### Phase/Milestone Grouping

Phases are numbered sequentially (1-10). Each has:
- A descriptive title
- LOE in days
- Optional **CR boundary** line: `**CR boundary:** ships on branch X, cut from commit Y.`
- Optional **Stage purpose** line: 1-2 sentences restating what the stage does.

### Definition-of-Done

Stated in a final section (`Notes`) or as a bold block at the end:

```
**Handoff completion definition.** The project is complete and handoff-ready when:
- <criterion>
- <criterion>
```

### Quantitative Density

- ~5,291 words total
- 10 phases (top-level tasks)
- 92 sub-tasks (checkboxes)
- Avg ~9 sub-tasks per phase
- Each sub-task: 2-5 lines including bullets and requirements reference

---

## 4. Key Design Decisions.md — Format and Conventions

### Section Hierarchy

```
# <Project> — Key Design Decisions (<Milestone>)    (H1)
> <One-line milestone description>                   (blockquote)
---
## N. <Decision title> (<Decision IDs if applicable>)    (H2)
```

### Decision Structure

Each decision has exactly four sections (bold labels, no headings):

```
## N. <Title>

**Why do we want to do this?**
<1-2 paragraphs: the problem or opportunity>

**How do we accomplish this?**
<1-3 paragraphs: the technical approach, specific enough to implement>

**What will replace this process?**
<1 sentence: what the old process was>

**Change to Process**
- **Use Case N — <scenario title>.**
  - Before: <how it worked before>
  - After: <how it works now>
```

### Conventions

- Decisions are numbered sequentially: `## 1.`, `## 2.`, etc.
- Parenthetical decision IDs reference external tracking: `(D5, D16)`
- Superseded decisions include a blockquote at the top: `> **Superseded by Decision N.**`
- Cross-references to other decisions: "Decision N" or "see Decision N"
- Bold inline for emphasis on key constraints: `**never**`, `**no**`
- Use Cases always have a Before/After pair, indented under a bold scenario title

### Level of Detail

**Very high.** Decisions name:
- Specific files/modules to be deleted or created
- Model names and versions
- Cost figures and ratios
- Configuration variable names
- Specific failure modes observed in practice

### Quantitative Density

- ~9,737 words total
- 25 decisions
- Avg ~390 words per decision
- 1-3 use cases per decision
- Some decisions include tables (model hierarchy, axis bindings)

---

## 5. Cross-Cutting Formatting Conventions

### Text Formatting

- **Bold** for: key terms on first use, section labels (Why/How/What/Change), emphasis on constraints
- *Italics* for: requirements references in tasks (`_Requirements: ..._`), property text in design
- `Code spans` for: file paths, function names, type names, enum values, config keys, API operations
- Blockquotes (`>`) for: supersession notices, addenda, parenthetical context

### Horizontal Rules

`---` separates requirement buckets and individual decisions. NOT used between sections.

### Code Fences

- Mermaid diagrams: ` ```mermaid `
- Data model pseudo-schemas: plain ` ``` ` (no language tag)
- ASCII dependency graphs: ` ```text `
- No language-tagged code blocks for implementation code in these spec docs

### Lists

- Numbered lists (`1.`) for acceptance criteria and sequential items
- Bullet lists (`-`) for glossary terms, implementation details, notes
- Nested bullets for sub-points, indented 2 spaces

### Cross-Document Consistency

- The same glossary terms appear across all four documents
- Requirements IDs are referenced identically in design (Correctness Properties) and tasks (italic references)
- Standing Decisions section appears in both requirements.md and design.md (identical content)
- Task phase numbers map to component/stage numbers in design.md

---

## 6. Verbatim Example Excerpts

### Example A — A Requirement with Acceptance Criteria (from requirements.md)

```markdown
### Requirement 5: Deterministic Structural Analyzers (Evidence-Only Findings)

**User Story:** As a region build engineer, I want structural analyzers that report plain,
repeatable evidence, so that the fact layer remains trustworthy and free of premature judgment.

**Requirement:** Run deterministic analyzers that emit only evidence. Each Finding is one of
`STRUCTURAL_LONGEST_PATH`, `SERIAL_CANDIDATE`, `MANUAL_TASK`, or `FAN_IN_HOTSPOT`, and carries
no verdict, recommendation, or severity.

#### Acceptance Criteria

1. WHEN the Structural_Analyzer runs over a Task_Graph, THE Structural_Analyzer SHALL emit Findings
   of the supported structural types.
2. WHEN the Structural_Analyzer emits a Finding, THE Structural_Analyzer SHALL populate only id,
   type, dimension, subjects, and evidence and SHALL NOT populate verdict, recommendation, or
   severity.
3. WHEN the Structural_Analyzer emits a `SERIAL_CANDIDATE`, THE Structural_Analyzer SHALL label it
   as a candidate and SHALL NOT assert that a dependency is removable.
4. WHEN the Structural_Analyzer emits a `MANUAL_TASK`, THE Structural_Analyzer SHALL record manual
   status as a fact and SHALL NOT prescribe automation.
5. WHEN the Structural_Analyzer runs twice over the same Snapshot, THE Structural_Analyzer SHALL
   produce identical Findings.
```

### Example B — A Task Entry (from tasks.md)

```markdown
- [x] 4.1 Implement the four evidence-only analyzers
  - Implement deterministic pure-function analyzers emitting `STRUCTURAL_LONGEST_PATH`, `SERIAL_CANDIDATE`, `MANUAL_TASK`, `FAN_IN_HOTSPOT` (task graph); populate only id/type/dimension/subjects/evidence with no verdict/recommendation/severity; label candidates and facts without prescribing removal or automation; `STRUCTURAL_LONGEST_PATH` is a code-derived structural signal (not RMS's authoritative critical path) and its evidence names RMS as the authoritative critical-path/slack source
  - _Requirements: 5.1, 5.2, 5.3, 5.4_
```

### Example C — A Data Model from design.md

```markdown
### Deviation record

Produced by stage C. An enumerable set of per-provider records consumed by stage D adjudication.

\```
DeviationRecord {
  provider_identity: string               // the declared or code-derived provider
  verdict:           MATCHED | DECLARED_ONLY | CODE_ONLY | UNRESOLVED
  rationale:         string               // judge-authored explanation
  citations:         Citation[]           // at least one code-corpus citation
  confidence:        string               // judge-assessed confidence level
}
\```
```

### Example D — A Design Decision (from Key Design Decisions.md)

```markdown
## 9. The Graph Builder is a thin, faithful task-graph projection that recomputes nothing and drops nothing (Task 3; Component 2; applies Decisions 1 and 5)

**Why do we want to do this?**
The Graph Builder sits between ingestion and analysis, and the tempting move is to make it "smart" —
verify the DAG, collapse tasks into their owning services, or prune dangling edges to tidy the
graph. Every one of those would either duplicate a guarantee RMS already provides or reintroduce the
exact silent data loss this milestone exists to eliminate.

**How do we accomplish this?**
The builder does exactly one thing — project a single immutable Snapshot onto the
backend-agnostic graph contract, one dimension at a time. It runs **no cycle detection**, performs
**no service collapse**, and **retains every edge, including dangling ones**.

**What will replace this process?**
A hand-read of the RMS graph — where an engineer mentally reconstructs the task graph and may quietly
skip an edge that points at something unfamiliar — is replaced by a deterministic projection that
keeps every element and fails loudly on the one thing that must never happen: two dimensions mixing.

**Change to Process**
- **Use Case 1 — an edge points at a task that wasn't loaded.**
  - Before: the reviewer skips the odd edge by eye, and the gap never surfaces.
  - After: the builder keeps the dangling edge in the graph and ingestion's discrepancy report
    already flagged it; nothing is silently tidied away.
```

### Example E — A Correctness Property from design.md

```markdown
### Property 3: No silent edge drops

*For any* ingested RMS graph, pagination is driven to exhaustion (guarded against a repeating-token
loop) so no page is silently dropped, and every dangling edge (an edge missing an endpoint) appears
flagged in the discrepancy report and is retained — no element is ever silently removed.

**Validates: Requirements 2.1, 2.2, 2.3**
```

---

## 7. Summary Table of Quantitative Targets

| Document | Words | Major Sections | Key Items | Avg Detail per Item |
|---|---|---|---|---|
| requirements.md | ~4,500 | 5 buckets | 11 requirements | 8.4 acceptance criteria each |
| design.md | ~7,000 | 7 components + data models + properties | 21 properties, 8 schemas | 3-5 sentences per property |
| tasks.md | ~5,300 | 10 phases | 92 sub-tasks | 2-5 lines per sub-task |
| Key Design Decisions.md | ~9,700 | 25 decisions | 25 decisions | ~390 words, 1-3 use cases |
| **TOTAL** | **~26,500** | — | — | — |

Writers should aim within ±20% of these word counts for equivalent-scope modules.
