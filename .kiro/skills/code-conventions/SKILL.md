---
name: ufcpolyml-code-conventions
description: >-
  Project code conventions for the UFC Polymarket workspace covering style, structure,
  naming, commenting, and test guidelines. Use when writing new code, modifying
  existing code, reviewing code changes, adding comments or docstrings, renaming
  symbols, creating new modules or classes, or refactoring. Do NOT use for prose
  writing, spec documents, build commands, or deployment tasks.
---

# Code Conventions

## General principles

- Follow standard conventions already established in this codebase unless they override what is in this document.
- Keep it simple. Simpler is always better. Reduce complexity as much as possible.
- Boy scout rule. Leave the code in the scope of your task cleaner than you found it.
- Always find root cause. Fix the source of a problem, not its symptoms.

## Design

- Keep configurable data at high levels.
- Prefer polymorphism to if/else or switch/case chains.
- Separate multi-threading code from business logic.
- Prevent over-configurability. Do not add options nobody asked for.
- Use dependency injection. Inject external clients; use deterministic fakes in tests.
- Follow Law of Demeter. A class should know only its direct dependencies.

## Code organization (project-specific)

- Give each behavioral class its own module, named after the class in `snake_case`: controllers, services, and ports. Plain data types — dataclasses, enums, typed records, exceptions — do not earn their own module; group them with the component they serve or in a shared `*_models.py` or `errors.py`. See `checker/models.py` and `population/report.py`. Modules of pure functions need no class at all and are named for what they do, such as `safe_path.py`.
- Use controller → service → port/util layering. Controllers orchestrate; services own one unit of work; ports define injected interfaces; utilities are pure and stateless.
- Dependencies point downward. Services and utilities must not import controllers, and ports must not import concrete services.
- Do not use module-level singletons or hidden global caches.
- Keep tests parallel to the source tree.
- `__init__.py` should not have a `__all__` section. Look at how other `__init__.py` files are done in this workspace without it.

## Source code structure

- Separate concepts vertically.
- Related code should appear vertically dense.
- Declare variables close to their usage.
- Dependent functions should be close.
- Similar functions should be close.
- Place functions in the downward direction (caller above callee).
- Keep lines short.
- Don't use horizontal alignment.
- Use white space to associate related things and disassociate weakly related.
- Don't break indentation.

## Naming

- Choose descriptive and unambiguous names.
- Make meaningful distinctions between similar names.
- Use pronounceable names.
- Use searchable names.
- Replace magic numbers with named constants.
- Avoid encodings. Don't append prefixes or type information.

## Functions

- Small. Each function does one thing.
- Use descriptive names.
- Prefer fewer arguments.
- Have no side effects.
- Don't use flag arguments. Split into independent methods the caller picks from directly.

## Objects and data structures

- Hide internal structure.
- Prefer data structures for pure-data types.
- Avoid hybrid structures (half object, half data).
- Keep classes small with a small number of instance variables.
- Base class should know nothing about its derivatives.
- Better to have many functions than to pass a code value to select behavior.
- Prefer non-static methods to static methods.

## Comments and docstrings (project-specific)

- Always try to explain yourself in code first. Comments are for what code cannot say.
- Don't be redundant. Don't add obvious noise.
- Don't use closing brace comments.
- Don't comment out code. Just remove it.
- Use comments as explanation of intent or clarification of non-obvious logic.
- Use comments as warning of consequences when relevant.

### File and class headers

Add an overview comment describing the file or class purpose and, when useful, its relationship to the containing folder or package. Keep it to 5–6 lines or fewer.

### Function headers

Briefly describe behavior, parameter and return types and purposes, preconditions or assumptions, and handled errors. Keep the header to 3 lines or fewer; use a compact format such as `Behavior/params/return:` and `Assumes/errors:`.

### Inline comments

Use sparingly for complex or unusual logic. Explain what the code accomplishes rather than repeating what it says; normally one line, rarely two. Avoid verbose and repetitive wording; keep comments direct and useful.

### Prohibited references

- Do not reference spec task numbers, requirement numbers, or property numbers in code, comments, docstrings, or test names. Reviewers outside the design have no way to resolve `Requirement 4.9`, `task 1.7`, or `Property 3`.
- Keep spec traceability in `.kiro/specs/<spec>/`, not in shipped source.
- Existing `Task`/`Property` tags in `tests/` (including the `property_tag()` helper in `tests/conftest.py`) are grandfathered — do not refactor them unless asked.

## Tests

- One assert per test when practical.
- Readable. A test should read like a short specification.
- Fast. Tests must not wait on I/O or sleep.
- Independent. No test relies on another test's state.
- Repeatable. Same result every run regardless of environment.

## Code smells to avoid

- **Rigidity** — a small change forces a cascade of subsequent changes.
- **Fragility** — the software breaks in many places due to a single change.
- **Immobility** — parts cannot be reused because of entanglement.
- **Needless complexity** — anticipatory design that serves no current requirement.
- **Needless repetition** — duplicated logic that should be unified.
- **Opacity** — code that is hard to understand without extensive study.

## Understandability

- Be consistent. If you do something a certain way, do all similar things the same way.
- Use explanatory variables.
- Encapsulate boundary conditions. Put boundary processing in one place.
- Prefer dedicated value objects to primitive types.
- Avoid logical dependency. Don't write methods that work correctly only because of state elsewhere in the same class.
- Avoid negative conditionals.
