# Git Workflow Guide

This document provides quick Git commands for committing changes to the repository.

## What Gets Committed

✅ **Tracked files (safe to commit):**
- Source code: `src/`, `scripts/`, `transforms/`
- Configurations: `configs/*.yaml`, `configs/README.md`
- Documentation: `README.md`, `TRAINING.md`, `QUICKSTART.md`, etc.
- Project files: `pyproject.toml`, `.gitignore`
- Notebooks: `notebooks/*.ipynb`

❌ **Ignored files (not committed):**
- Training outputs: `outputs/`
- Model checkpoints: `models/`
- Dataset: `data/`
- Python cache: `__pycache__/`, `*.pyc`
- Virtual environments: `.venv/`, `venv/`
- IDE settings: `.vscode/` (if configured)

## Basic Git Workflow

### 1. Check Current Status

```bash
git status
```

This shows:
- Modified files (red = unstaged, green = staged)
- Untracked files
- Current branch

### 2. Stage Changes

```bash
# Stage specific files
git add configs/baseline_cnn_no_augmentation.yaml
git add README.md TRAINING.md

# Or stage all tracked changes
git add -u

# Or stage everything (be careful with new files!)
git add .
```

### 3. Commit Changes

```bash
# Commit with descriptive message
git commit -m "Add baseline_cnn_no_augmentation config for pure CNN baseline"

# Or multi-line commit message
git commit -m "Add pure baseline configuration

- Create baseline_cnn_no_augmentation.yaml (30 epochs, no augmentation)
- Rename baseline.yaml to baseline_cnn_augmented.yaml
- Update documentation to reflect multiple baselines
- Add configs/README.md to document all configurations"
```

### 4. Push to GitHub

```bash
# Push to main branch
git push origin main

# Or if tracking is already set up
git push
```

## Quick Workflow for Current Changes

For the baseline configurations just created:

```bash
# 1. Check what changed
git status

# 2. Stage the new baseline files
git add configs/baseline_cnn_no_augmentation.yaml
git add configs/baseline_cnn_augmented.yaml
git add configs/README.md
git add transforms/__init__.py
git add README.md
git add TRAINING.md

# 3. Commit with clear message
git commit -m "Add pure baseline CNN configuration (first baseline)

- Add baseline_cnn_no_augmentation.yaml: 30 epochs, no augmentation
- Rename baseline.yaml → baseline_cnn_augmented.yaml for clarity
- Update transforms to support no-augmentation mode
- Document multiple baseline configurations in README and TRAINING
- Add configs/README.md to explain baseline strategy"

# 4. Push to GitHub
git push origin main
```

## View Commit History

```bash
# View recent commits
git log --oneline -10

# View detailed history
git log

# View changes in last commit
git show
```

## Undo Mistakes (Before Push)

```bash
# Unstage files (keep changes)
git restore --staged <file>

# Discard local changes (careful!)
git restore <file>

# Amend last commit (if not pushed yet)
git commit --amend -m "New commit message"
```

## Branch Management (Optional)

If you want to work on a feature branch:

```bash
# Create and switch to new branch
git checkout -b feature/new-baseline

# Make changes and commit
git add .
git commit -m "Add new baseline"

# Push branch to GitHub
git push origin feature/new-baseline

# Merge to main later
git checkout main
git merge feature/new-baseline
git push origin main
```

## Best Practices

1. **Commit often**: Small, focused commits are better than large ones
2. **Clear messages**: Describe what and why, not just "updates"
3. **Check status**: Always run `git status` before committing
4. **Review changes**: Use `git diff` to review before staging
5. **Pull before push**: If working with others, pull latest changes first

## Common Issues

### Merge Conflicts

```bash
# If you have conflicts after pulling
git pull origin main

# Fix conflicts in files (look for <<< === >>> markers)
# Then:
git add <fixed-files>
git commit -m "Resolve merge conflicts"
git push origin main
```

### Accidentally Staged Wrong Files

```bash
# Unstage specific file
git restore --staged <file>

# Unstage all files
git restore --staged .
```

### Need to See What Changed

```bash
# See unstaged changes
git diff

# See staged changes
git diff --cached

# See changes in specific file
git diff <file>
```

---

**Ready to commit your baseline configurations?** Use the "Quick Workflow" section above!
