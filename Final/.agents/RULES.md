# Project Rules & Guidelines

## Strict Directory Boundary
- All edits must happen strictly inside the `Harshit` folder.
- Only reading and copying from the root workspace directory is allowed. No files outside `Harshit` may be created or modified.

## Python Execution Rules
- Always use the local virtual environment `Harshit\.venv` (`uv` or `Harshit\.venv\Scripts\python.exe`).
- PowerShell syntax only: use `;` for chaining, not `&&`.

## Trading & Strategy Rules (from GEMINI_INSTRUCTIONS.md & Rulebook)
1. Never edit `engine.py` or `compare.py`.
2. `strategy.py` may import only `numpy`, `pandas`, `sklearn` and the Python standard library. No file reading and no internet inside `strategy.py`.
3. Never use day numbers in any `decide` function. Do not read the values of `history["close"].index`, do not use `%` on day numbers, do not compare to specific days. Use only `.iloc` positions and price values.
4. Every function whose name starts with `decide_` must take exactly one argument (`history`) and return a dict `{stock_name: weight}`. Helper functions must start with an underscore (`_`), never with `decide_`.
5. Do not delete `decide_momentum`, `decide_trend_filter`, or the existing lines inside `train()`.
6. All weights must be >= 0 (no shorting) and their sum must be <= 1.
7. Never write `MODEL = {...}` inside a function. Only set keys: `MODEL["name"] = value`.
8. Do not change `TRAIN_WINDOW` until Step 7 instructs.
9. Never invent numbers for the README or anywhere else. Copy numbers only from the CSV files.
10. Do not put `print` calls inside any `decide_` function or helper.
11. When a step says STOP, stop and wait for user instructions.

