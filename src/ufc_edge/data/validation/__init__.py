"""Self-consistency validation suite for scraped ufcstats data.

Replaces the dropped Kaggle cross-check (Key Design Decisions D8). Invariants
detect *impossible* data (parser bugs, source corruption) — never *incomplete*
data, which the Missingness Policy handles. Violating rows are quarantined with
reason codes and excluded from feature computation; alarms are era-scoped per
D12: any violation in the label universe (event_date >= label_start_date, D11)
fails the suite, pre-cutoff violations are report-only.
"""
