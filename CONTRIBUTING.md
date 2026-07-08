# Contributing

Thanks for your interest in improving Continuous-time Causal Prior-Fitted
Networks. Contributions (bug reports, fixes, and improvements) are welcome.

## Development setup

```bash
git clone https://github.com/thummd/continuous-time-causal-pfn
cd continuous-time-causal-pfn
python -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis]"     # analysis extra pulls in the baseline deps
pytest tests/                    # sanity check
```

The temporal-SCM prior (`causal_time_prior/`) and `dopfnprior/` are vendored at
the repository root, so no external repositories are required.

## Tests

Run the test suite after any change that touches shared code:

```bash
pytest tests/
```

Please add or update tests for behaviour you change.

## Pull requests

1. Open an issue first for anything substantial, so we can agree on the approach.
2. Keep PRs focused; write a clear description of what changed and why.
3. Ensure `pytest tests/` passes.
4. Match the surrounding code style.

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache-2.0](LICENSE) license.
