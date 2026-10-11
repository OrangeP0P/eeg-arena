# Results Dashboard

Build a static HTML results site from one or more batch output directories:

```bash
python results/build_site.py \
  --suite-dir experiments/output/smoke_suite
```

Optional arguments:

- `--site-name <name>`: override the site name and default output leaf directory.
- `--output-dir <path>`: write artifacts to an exact directory.
- repeat `--suite-dir` to merge multiple suites with deduplicated condition-level summaries.

Outputs:

- `dashboard.html`
- `dashboard_data.json`
- `raw_runs.csv`
- `effective_runs.csv`
- `metadata.json`

Default output location:

```text
results/output/<site-name>/
```
