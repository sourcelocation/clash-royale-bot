# C++ Backend Quick Commands

Run from repo root.

## Build
```bash
./cpp/compile.sh
```

## Validate Environment (single-env smoke)
```bash
python -m training.custom.validate_env --steps 100 --tick-hz 10 --max-sim-seconds 120
```

## Tests (fast)
```bash
pytest training/tests/cpp_backend -q
```

## Tests (slow)
```bash
pytest training/tests/cpp_backend -m slow -q
```

## Visualizer
```bash
python -m training.tools.cpp_visualizer --fps 60 --decision-hz 1 --max-sim-seconds 120
```

## Visualizer (limited step rate)
```bash
python -m training.tools.cpp_visualizer --fps 60 --limit-steps-per-second 1000 --decision-hz 1
```

## Training (cpp-only)
```bash
./godot/train.sh
```

Or explicit:
```bash
python -m training.custom.train_ppo --num-envs 5 --cpp-tick-hz 10 --cpp-num-threads 0
```
