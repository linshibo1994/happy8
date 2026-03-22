# Repository Guidelines

## Project Structure & Module Organization
This repository contains two active codebases plus planning docs:
- `src/`, `main.py`, `data/`, `deployment/`, `docs/`: core Happy8 Python prediction system (Streamlit + CLI).
- `happy8-miniprogram/frontend/`: uni-app + Vue 3 + TypeScript mini-program client.
- `happy8-miniprogram/backend/`: FastAPI backend (`app/api`, `app/services`, `app/models`, `app/core`).
- `happy8-miniprogram-specs/`: requirements, design, and task breakdown.

Keep business logic in service layers (`src/*` or `backend/app/services/*`), not in route or page files.

## Build, Test, and Development Commands
Root (Python prediction app):
```bash
pip install -r requirements.txt
python main.py web      # start Streamlit UI
python main.py demo     # run demo prediction flow
python main.py deploy   # run deployment checks/scripts
```

Mini-program backend:
```bash
cd happy8-miniprogram/backend
pip install -r requirements.txt
python start.py         # run FastAPI service
```

Mini-program frontend:
```bash
cd happy8-miniprogram/frontend
npm install
npm run dev:mp-weixin   # WeChat mini-program dev build
npm run dev:h5          # H5 local preview
npm run build:mp-weixin
npm run lint && npm run type-check
```

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes.
- FastAPI layering: API in `app/api/v1`, business logic in `app/services`, models in `app/models`.
- Vue/TS: follow ESLint + Prettier (`singleQuote: true`, `semi: false`, `tabWidth: 2`, `printWidth: 100`).
- Component files use `PascalCase` (e.g., `NumberBall.vue`); page directories use lowercase by feature (`pages/predict/`, `pages/profile/`).

## Testing Guidelines
- Backend test stack: `pytest`, `pytest-asyncio`, `pytest-cov`.
- Place tests in `happy8-miniprogram/backend/tests/` with `test_*.py` naming.
- Run:
```bash
cd happy8-miniprogram/backend
pytest -q
pytest --cov=app --cov-report=term-missing
python test_app.py      # quick import/config smoke check
```

## Commit & Pull Request Guidelines
- Match existing history style: concise Chinese summaries, optional emoji prefix (e.g., `🔧 修复...`, `🚀 新增...`).
- Keep one logical change per commit; avoid mixing frontend/backend refactors.
- PRs should include: scope, affected paths, verification commands run, and screenshots for UI changes (`frontend/src/pages/*`).
- Link related issue/spec entries when applicable.

## Security & Configuration Tips
- Do not commit secrets (`.env`, API keys, payment credentials).
- Use `happy8-miniprogram/backend/.env.example` as the baseline for local config.
- Validate API base URL and WeChat app settings before production builds.
