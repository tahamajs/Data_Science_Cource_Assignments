# Contributing

## Development Workflow

1. Create a feature branch.
2. Install project dependencies.
3. Run tests before opening a PR.
4. Keep changes focused and document behavior changes.

## Local Commands

```bash
make install
make test
make run
make compile
```

## Code Standards

- Keep functions small and deterministic where practical.
- Avoid data leakage in all training pipelines.
- Preserve reproducibility via explicit random seeds.
- Update docs when outputs or interfaces change.
