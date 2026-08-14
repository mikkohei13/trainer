See `ARCHITECTURE.md` for system architecture.

## Development principles

- Prefer simple, clear solutions. Avoid unnecessary abstractions, fallbacks, and defensive complexity.
- Keep changes focused on the requested task. Do not add unrelated features or infrastructure.
- Put application and business logic in the Python backend. Use JavaScript only for UI enhancements; do not move core application logic to the frontend.
- Do not use heavy JavaScript frameworks.
- When changing the database schema, do the required migration. Backward compatibility with old database schemas is not required.
- Add unit tests in `./tests` for important behavior only.

## Product constraints

This is a desktop-only application maintained and used by one person only. Keep the implementation accordingly simple.

- No authentication.
- No mobile support; optimize the UI for desktop.
- Do not add accessibility-specific features.
- Avoid premature optimization, unnecessary scalability work, and multi-user architecture.