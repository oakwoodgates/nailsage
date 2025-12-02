# Deprecated Scripts

This folder contains old strategy-specific training and validation scripts that have been replaced by generic, configuration-driven scripts.

## Deprecated Scripts

### train_momentum_classifier.py
**Replaced by:** `scripts/train_model.py`

**Why deprecated:**
- Strategy-specific implementation made it difficult to scale to 100s of models
- Required duplicating code for each new strategy
- Configuration was hardcoded in the script

**Migration:**
Use the new generic training script with strategy config files:
```bash
python scripts/train_model.py --config configs/strategies/your_strategy.yaml
```

### validate_momentum_classifier.py
**Replaced by:** `scripts/validate_model.py`

**Why deprecated:**
- Strategy-specific implementation for walk-forward validation
- Required separate script for each strategy type
- Combined training and validation in one file

**Migration:**
Use the new generic validation script:
```bash
python scripts/validate_model.py --config configs/strategies/your_strategy.yaml --model-id YOUR_MODEL_ID
```

## Architecture Change

**Old Approach:**
- Strategy-specific scripts in `/strategies/short_term/`, `/strategies/long_term/`, etc.
- Each strategy required its own training and validation scripts
- Bug fixes required updating multiple files
- Configuration mixed with code

**New Approach:**
- Single generic scripts in `/scripts/` that work with any strategy
- All strategy differences are in YAML configuration files
- Bug fixes apply to all strategies automatically
- Clear separation of configuration and code
- Better maintainability at scale

## When to Use Old Scripts

These scripts are kept for reference only and should not be used for new work. They contain useful logic that was ported to the new generic scripts.

If you need to reference how a specific feature worked, check these files, but implement it in the new generic scripts.

## Date Deprecated

2025-11-26
