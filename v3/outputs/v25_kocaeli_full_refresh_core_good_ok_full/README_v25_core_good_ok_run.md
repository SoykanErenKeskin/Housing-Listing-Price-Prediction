# V25 Kocaeli Full Refresh — core_good_ok

- model_version: `v25-kocaeli-full-refresh-core-good-ok`
- model_scope: `kocaeli_global`
- county_policy: `core_good_ok`
- included: Darıca, İzmit, Gebze, Başiskele, Çayırova, Gölcük, Kartepe, Körfez, Derince
- excluded WEAK: Karamürsel, Kandıra, Dilovası
- selected_experiment: `full_v25_core_good_ok`
- best_checkpoint: `True`
- promote_over_v24_1: `false`

## Same-data control
- control R2=0.7307693679783922 MAPE=0.1082550641142819
- full_v25 R2=0.7307693679783922 MAPE=0.10825506411428192

## Old V24.1 reference (different inventory distribution)
- R2=0.6523243261980677 MAPE=0.11762822096027113

Do not update EDER/app artifacts from this run.
