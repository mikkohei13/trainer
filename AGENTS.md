See ARCHITECTURE.md for the system architecture.

## Development principles

- Keep it simple and clear.
- No need for complex fallback or error messaging logic.
- If you cnage database structure, do necessary migrations. But there is no need for backwards compatibility for old database schemas.
- Add unit tests to ./tests for the most important features only.
-  This is one-person app, so avoid premature optimization and unnecessary complexity:
  - No accessibility features
  - No authentication
  - Desktop-optimized UI, no mobile support
  - No heavy JavaScript frameworks
- Favor creating app/business logic in Python backend over JavaScript frontend. This is a backend app, with JavaScript only used to improve the UI, not to drive the app itself.