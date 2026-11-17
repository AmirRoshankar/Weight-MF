# Reproducing Weight-MF

The README is the research narrative. This file specifies the executable experiments and distinguishes recorded measurements from newly generated outputs.

## Layout

- src/grokflow/: source transformer, checkpoint datasets, block/embedding generators and modular-task evaluation.
- experiments/: three command-line experiment implementations; curriculum/ contains the five-block architecture and the paired curriculum treatments.
- scripts/run.py: public entry point. New runs go to runs/, never to the recorded results.
- results/: tabular measurements, per-seed stage curves, and source/generator configurations.
- figures/: generated from results by scripts/plot_results.py.
- configs/: source splits and generator settings.

The code retains the historical numerical operations while removing unrelated task trainers and orchestration. The helper named experiments/curriculum/train.py contains only model construction, generator loading, and normalization.

## Environment

Use Python 3.11 or newer and a CUDA-compatible PyTorch 2.9 installation. Historical runs used Python 3.13 and CUDA PyTorch 2.9.1. Install with:

    python -m pip install -e .
    python scripts/run.py --help
    python -m unittest discover -s tests

Curriculum experiments require CUDA. Import/model tests and figure regeneration run on CPU.

## Figures and tables

    python scripts/run.py plots

This regenerates every README figure from the retained CSVs, including every individual modular operation. Bands denote one standard deviation over training seeds. They do not represent uncertainty over independent datasets. Source-scaling and modular threshold means both censor failures at 20,000 updates. Historical scaling_summary.csv also retains its original success-conditional means; censor_aware_summary.csv is the comparable statistic used by the README. Modular accuracy curves carry the last observation forward after a run stops; those plateaus are not further optimizer updates.

## Source corpus and generator

    python scripts/run.py collect-source
    python scripts/run.py train-generator --source transition

Collection replays source seeds 20000 through 20601 and keeps ten transition-focused pairs per successful trajectory. The generator uses the original 500/51/51 whole-run split and 10,000 training updates. If a specified source seed fails to grok, collection stops instead of substituting a new seed. Outputs are data/source_transition and artifacts/transition.

    python scripts/run.py recollect-source
    python scripts/run.py train-generator --source stratified

Recollection requires the transition corpus first. It replays the same seeds and spans, matches each run's ten pair gaps, and draws starts from feasible-time deciles. The source split is in configs/transition/split_manifest.json. Outputs are data/source_stratified and artifacts/stratified.

Generator checkpoints and large corpora are generated locally and ignored by Git. There is no assumed private download location. Reproducing full training requires the collection/training commands above; these are computational experiments, not an instant evaluation script.

## Modular-operation transfer

    python scripts/run.py modular

Uses artifacts/transition/best_model.pt, the eight operation/modulus combinations, and paired seeds 31000-31019. The six matrices are replaced once at step 500 with endpoint prompt 9000 and corresponding AdamW state cleared. Division uses modular inverses over nonzero divisors; XOR labels are reduced modulo the output vocabulary. The input is three categorical tokens, not digit-level text.

Optional original CLI flags can follow the public arguments, for example:

    python scripts/run.py modular --output runs/modular_check --task multiplication:113 --trials 1

## Curriculum, attribution and horizons

    python scripts/run.py curriculum
    python scripts/run.py controls
    python scripts/run.py source-sampling
    python scripts/run.py horizons

The first command trains optimizer and 0.1 iMF treatments for seeds 101-105. The control command tests reset-only, iMF with reset, iMF preserving state, and Gaussian random directions matched per matrix to the contemporaneous applied 0.1 proposal. The last two commands require the stratified generator. Horizon comparisons use +100 and +200 against the original min(30000,max(9000,start+1000)) prompt rule.

Each stage runs 20,000 AdamW updates. Stage 1 learns elementary operations. Stage 2 starts from the minimum-validation-loss Stage 1 model with a fresh optimizer. Both stages supervise only the answer token. Stage 2 batches contain 115 composed and 13 elementary examples. Interventions occur before the next optimizer update at local steps 0,500,...,19500. Strength, intervention frequency and endpoint horizon are distinct parameters. See the source code and configs for the full protocol.

Generated datasets are disjoint by expression within train/validation/holdout. Dataset generation is deterministic with its fixed seed. Model/minibatch/noise pairing uses the same historical seed formulas. Five training seeds on one target dataset do not measure independent-dataset variability.

Choose a new --output directory for independent reruns; the historical runner resumes saved checkpoints and skips completed summaries within that output. It never treats the bundled results as a new execution. Interrupted source runs restart that trajectory; completed shards are reused.

## Embedding-only source-data experiment

    python scripts/run.py collect-embedding
    python scripts/run.py source-scaling

The historical split was drawn from 1,102 source trajectories (seeds 100-1201); preserving that candidate pool is necessary even when only 50,150,500 trajectories train each predictor. The scaling command trains predictors for 30,600 updates, uses fixed source validation/test memberships and twenty downstream seeds, and evaluates the 50/50 and 10/90 mixtures. This embedding-only velocity regression protocol differs from the later block iMF.

## Verification and limits

Reorganization checks exercise imports, model-copy equivalence, six-matrix round trips, dataset labels/splits, CLI dispatch, CSV-based summaries and figure regeneration. These checks do not rerun all historical GPU trials. Exact numerical reproduction across different devices or library versions is not guaranteed. The original same-host seed checks and source replay audit support the recorded execution, rather than a general cross-platform guarantee.

All selected results are experimental measurements. Endpoints at equal Stage 2 budget differ from validation-loss-selected checkpoints and from time-to-threshold metrics. Source collection and generator cost must be included before claiming end-to-end compute savings.
