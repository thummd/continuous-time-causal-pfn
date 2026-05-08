# Contributing to continuous-time-causal-pfn

This is a private fork of [do-over-time-pfn](https://github.com/thummd/do-over-time-pfn)
dedicated to the ICML FMSD 2026 workshop paper on continuous-time causal
PFNs. This file documents the workflow.

## Onboarding (new collaborator)

```bash
# 1. Clone without LFS blobs (the fork does not store them — they live upstream)
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:thummd/continuous-time-causal-pfn.git
cd continuous-time-causal-pfn

# 2. Register the upstream DoT-PFN repo
git remote add upstream git@github.com:thummd/do-over-time-pfn.git
git fetch upstream

# 3. Configure LFS to skip fetching by default
git config lfs.fetchexclude "*"
git config lfs.allowincompletepush true

# 4. Check out the workshop development branch
git checkout ct-dev

# 5. Install the environment
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest tests/   # sanity check
```

## Branch layout

| Branch | Purpose |
|---|---|
| `main` | Stable. Tracks DoT-PFN `main` after periodic syncs. |
| `dennis` | Mirror of upstream `dennis` (latest DoT-PFN work). Do **not** commit workshop-specific changes here. |
| `ct-dev` | Workshop development trunk. Branch feature branches off this. |
| `ct-dev/<topic>` | Feature branches (e.g. `ct-dev/sde-prior`, `ct-dev/dt-encoder`). Merge back into `ct-dev`. |

## Where code goes

```
dotime/prior/continuous/    <- SDE / OU samplers, Delta-t scheduling
dotime/model/continuous/    <- Delta-t aware encoder / mixer variants
dotime/data/pk_pd/          <- PK/PD benchmark loaders
paper/icml_fmsd/            <- workshop paper LaTeX
```

Keep continuous-time extensions inside the `continuous/` subdirectories.
Editing discrete-time code in the parent directories is allowed **only**
if the change is safe to upstream back to DoT-PFN.

## Syncing with upstream DoT-PFN (weekly)

```bash
git fetch upstream

# Pull upstream main into our main
git checkout main
git merge upstream/main
git push origin main

# Mirror upstream dennis (fast-forward only; do not commit locally)
git checkout dennis
git merge --ff-only upstream/dennis
git push origin dennis

# Fold upstream changes into the workshop branch
git checkout ct-dev
git merge main          # or: git merge dennis -- depending on what you want to track
pytest tests/           # run DoT-PFN tests after every sync
git push origin ct-dev
```

If a DoT-PFN refactor breaks continuous-time code, resolve conflicts
inside `dotime/*/continuous/` without touching the upstream files.

## Deploying to the GPU server

The shared GPU server (`aidf-svr-gpu04`) does **not** keep a `.git` checkout
of this repo — it runs from a plain working tree at
`~/repos/continuous-time-causal-pfn/`. Two ways to push code there:

1. **Targeted scp** (preferred for small edits): copy only the files you
   changed. Safe by construction, never touches `checkpoints/` or `wandb/`.

   ```bash
   scp dotime/training/continuous_trainer.py \
       dennis@10.230.252.6:repos/continuous-time-causal-pfn/dotime/training/
   ```

2. **rsync of the whole tree** (for many files at once): **always** use
   `--exclude-from=.rsyncignore` and **never** add `--delete` without
   reading the exclude list first. Otherwise you risk wiping training
   artefacts the server generated.

   ```bash
   rsync -avz --exclude-from=.rsyncignore \
       ./ dennis@10.230.252.6:repos/continuous-time-causal-pfn/
   ```

   The repo ships a top-level `.rsyncignore` that excludes
   `checkpoints/`, `results/`, `wandb/`, `logs/`, LaTeX intermediates, and
   editor caches. **This file is the canonical exclude list — read it
   before changing the rsync invocation.**

> **Postmortem note (May 2026):** Six of the eight `grid_v4` ablation
> checkpoints (~600 MB total) vanished between Apr 29 and May 1, 2026
> because a workspace sync ran without exclusions and `--delete`-style
> pruned the server's `checkpoints/` tree. The trainer now mirrors every
> best checkpoint to wandb via `wandb.save(..., policy="live")` so a
> future workspace wipe is recoverable.

## Upstreaming continuous-time work into DoT-PFN

When a continuous-time component is stable enough to ship in the NeurIPS
paper:

```bash
# In a separate clone of the upstream DoT-PFN repo
cd ~/repos/do-over-time-pfn
git fetch
git checkout -b continuous-time-prior origin/dennis

# Cherry-pick the relevant commits from the fork
git remote add ct-fork git@github.com:thummd/continuous-time-causal-pfn.git
git fetch ct-fork
git cherry-pick <commit-range-on-ct-fork>

git push origin continuous-time-prior
# Open a PR in github.com/thummd/do-over-time-pfn
```

Because continuous-time code is namespaced under `continuous/`, the
cherry-pick should rarely conflict with ongoing DoT-PFN work.

## Commit conventions

Workshop-specific commits should be prefixed so upstreaming later is
easier:

- `[ct-prior]` SDE / OU / Delta-t sampling changes
- `[ct-encoder]` Delta-t aware encoder modifications
- `[ct-eval]` PK/PD / CausalChamber continuous-time evaluation
- `[ct-paper]` workshop paper edits only
- `[sync]` merges from upstream

Commits touching only `paper/icml_fmsd/` should never be upstreamed;
commits touching only discrete-time code should always be candidates for
upstream.

## Running the existing DoT-PFN test suite

After any change that touches shared code, run the inherited tests:

```bash
pytest tests/
```

If a test fails and the failure is on discrete-time code, either fix it
or tag the commit with `[sync]` and open an issue upstream.

## LFS policy

- The fork does **not** store DoT-PFN's historical checkpoints. Pull them
  from upstream on demand: `git lfs pull upstream --include "<path>"`.
- New checkpoints from continuous-time experiments go into this fork's
  LFS. Unset `lfs.allowincompletepush` for commits that include new
  checkpoint files to ensure they actually upload.

## Contacts

- Repo owner: Dennis Thumm (`dennis.thumm@u.nus.edu`)
- Workshop: [ICML 2026 FMSD](https://icml-structured-fm-workshop.github.io/)
