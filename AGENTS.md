# Repository Guidelines

## Project Structure & Module Organization
This repository now follows a single-backend structure:
- `backend/`: canonical FastAPI backend (`app/api`, `app/services`, `app/models`, `app/core`, `tests`).
- `engine/`: Happy8 prediction engine and analysis scripts.
- `infra/`: database init files, Nginx config, and deployment scripts.
- `docs/`: project and backend documentation.
- `specs/`: requirements, design, and task breakdown.

Keep business logic in service layers (`engine/*` or `backend/app/services/*`), not in route files.

## Build, Test, and Development Commands
Root:
```bash
pip install -r requirements.txt
python main.py api      # start FastAPI backend
python main.py demo     # run demo prediction flow
```

Backend:
```bash
cd backend
pip install -r requirements.txt
python start.py
```

Docker:
```bash
docker compose up -d
docker compose logs -f backend
```

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes.
- FastAPI layering: API in `app/api/v1`, business logic in `app/services`, models in `app/models`.
- Algorithm code in `engine/` should remain framework-agnostic and reusable by the backend.

## Testing Guidelines
- Backend test stack: `pytest`, `pytest-asyncio`, `pytest-cov`.
- Place tests in `backend/tests/` with `test_*.py` naming.
- Run:
```bash
cd backend
pytest -q
pytest --cov=app --cov-report=term-missing
python test_app.py
```

## Commit & Pull Request Guidelines
- Match existing history style: concise Chinese summaries, optional emoji prefix.
- Keep one logical change per commit; avoid mixing engine, backend, and infra refactors when possible.
- PRs should include: scope, affected paths, and verification commands run.

## Security & Configuration Tips
- Do not commit secrets (`.env`, API keys, payment credentials).
- Use `backend/.env.example` as the baseline for local config.
- Validate database, Redis, and payment configuration before production builds.
