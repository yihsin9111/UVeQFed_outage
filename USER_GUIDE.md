# Refactored CIFAR-10 Federated Learning: User Guide

## Overview

The original `CIFAR.m` has been refactored into a clean, parameterized framework for running experiments with different:
- **Random seeds** (for statistical significance / error bars)
- **Dataset types** (IID and non-IID distributions)
- **Quantization schemes** (6 different methods)
- **Quantization rates** (compression levels)

---

## File Structure

### New Files
- **`run_cifar_experiment.m`** — Main parameterized experiment function
- **`run_batch_experiments.m`** — Batch runner for multiple configurations  
- **`QUICK_START.m`** — Copy-paste examples (no execution)
- **`investigate.md`** — Random seed investigation + usage notes

### Original Files (Unchanged)
- `CIFAR.m` — Original code (still works for reference)
- `GetUnbalancedCIFAR.m` — Non-IID dataset generator
- `m_fQuantizeData.m`, `m_fGenDither.m`, etc. — Supporting functions

---

## Quick Usage

### Option A: Single Experiment
```matlab
error = run_cifar_experiment('seed', 42, 'quant_type', 2, 'quant_rate', 4, 'proposed', 1);
```

Results automatically save to: `./results/2D_Lattice_4_iid_seed_42.mat`

### Option B: Multiple Seeds (Error Bars)
```matlab
all_errors = [];
for seed = 1:5
    error = run_cifar_experiment('seed', seed, 'quant_type', 2, 'proposed', 1);
    all_errors = [all_errors, error];
end

mean_acc = mean(all_errors, 2);
std_acc = std(all_errors, [], 2);
errorbar(1:40, mean_acc, std_acc);
```

### Option C: Full Batch (Recommended for Comprehensive Testing)
```matlab
run_batch_experiments();
```
- Runs **all** combinations of seeds, datasets, and quantization schemes
- Auto-generates comparison plots
- Saves everything to `./results/`

---

## Function Reference

### `run_cifar_experiment()`

**Syntax:**
```matlab
error_curve = run_cifar_experiment(Name, Value, ...)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 1 | Random seed (0 to 2^32-1) |
| `dataset_type` | string | `'iid'` | `'iid'` or `'non-iid'` |
| `ratio` | double | 0.5 | Non-IID ratio [0,1], only used if `dataset_type='non-iid'` |
| `quant_type` | int | 2 | Quantization type (1-6, see below) |
| `quant_rate` | double | 4 | Quantization bits/sample |
| `proposed` | int | 0 | 0=no quantization, 1=apply quantization |
| `save_results` | bool | true | Auto-save to `./results/` |

**Quantization Types:**
- `1` = 3D Lattice
- `2` = 2D Lattice ⭐ (recommended)
- `3` = Scalar
- `4` = QSGD
- `5` = Uniform with random unitary rotation
- `6` = Subsampling with 3-bit quantizers

**Output:**
- `error_curve` : 40×1 vector of accuracy across iterations
- File saved as: `./results/<scheme>_<rate>_<iid_type>_seed_<seed>.mat`

**Examples:**
```matlab
% Baseline IID, no quantization
run_cifar_experiment('seed', 1);

% IID with 2D lattice quantization
run_cifar_experiment('seed', 1, 'quant_type', 2, 'quant_rate', 4, 'proposed', 1);

% Non-IID (ratio=0.3) with QSGD
run_cifar_experiment(...
    'seed', 5, ...
    'dataset_type', 'non-iid', ...
    'ratio', 0.3, ...
    'quant_type', 4, ...
    'proposed', 1 ...
);
```

---

## Output Files

### Result File Naming
```
./results/<quantization_scheme>_<rate>_<dataset_type>_seed_<seed>.mat
```

**Examples:**
```
2D_Lattice_4_iid_seed_1.mat
2D_Lattice_4_iid_seed_2.mat
Scalar_2_non_iid_ratio_0.30_seed_1.mat
QSGD_4_non_iid_ratio_0.50_seed_3.mat
```

**File Contents:**
```matlab
load('2D_Lattice_4_iid_seed_1.mat');
% Loads: error_result (40×1 vector of accuracy values)
```

### Batch Summary File
When using `run_batch_experiments()`:
```
./results/experiment_summary.mat
```
Contains:
- `all_results{}` — Cell array of all experiment results with statistics
- `dataset_configs` — Configuration matrix
- `quant_configs` — Configuration matrix
- `seeds` — Seed values used

---

## Example Workflow: Generate Error Bars

### Step 1: Run multiple seeds
```matlab
num_seeds = 5;
quant_type = 2;
quant_rate = 4;

for seed = 1:num_seeds
    run_cifar_experiment(...
        'seed', seed, ...
        'quant_type', quant_type, ...
        'quant_rate', quant_rate, ...
        'proposed', 1 ...
    );
end
```

### Step 2: Load and analyze
```matlab
% Load all results
all_errors = [];
for seed = 1:num_seeds
    filename = sprintf('2D_Lattice_%g_iid_seed_%d.mat', quant_rate, seed);
    load(fullfile('./results', filename));
    all_errors = [all_errors, error_result];
end

% Calculate statistics
mean_acc = mean(all_errors, 2);
std_acc = std(all_errors, [], 2);
sem_acc = std_acc / sqrt(num_seeds);  % Standard error of mean

% Plot
figure;
iterations = 1:length(mean_acc);
errorbar(iterations, mean_acc, std_acc, 'o-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Accuracy');
title(sprintf('2D Lattice (rate=%g): %d seeds', quant_rate, num_seeds));
grid on;
```

### Step 3: Compare multiple schemes
```matlab
schemes = {
    {2, 4, 1},  % 2D Lattice
    {3, 4, 1},  % Scalar
    {4, 4, 1},  % QSGD
};

scheme_names = {'2D Lattice', 'Scalar', 'QSGD'};

figure; hold on;
for i = 1:length(schemes)
    quant_type = schemes{i}{1};
    quant_rate = schemes{i}{2};
    
    all_errors = [];
    for seed = 1:5
        % Load and concatenate...
    end
    
    mean_acc = mean(all_errors, 2);
    errorbar(iterations, mean_acc, std(all_errors, [], 2), 'o-', 'LineWidth', 2);
end

legend(scheme_names);
xlabel('Iteration');
ylabel('Accuracy');
title('Quantization Scheme Comparison');
grid on;
```

---

## Random Seed Setup (Behind the Scenes)

The function **automatically sets `rng(seed_val)` at the start**, which controls:

1. **NN weight initialization** in `trainNetwork()`
2. **Mini-batch shuffling** during SGD
3. **Data splitting** in `splitEachLabel()` and `GetUnbalancedCIFAR()`
4. **Dither generation** in quantization functions

All randomness is deterministic once seed is set, so **exact same results with same seed**.

---

## Tips & Tricks

### Faster Testing (Reduced Data)
Edit `run_cifar_experiment.m` line ~120:
```matlab
iteration = 10;  % Instead of 40 (faster testing)
```

### Custom Learning Rates
Edit lines ~150-180 in `run_cifar_experiment.m` to adjust `trainingOptions`.

### Disable Result Saving
```matlab
run_cifar_experiment('seed', 1, 'save_results', false);
```

### Check What's Being Tested
The function prints detailed info:
```
========== CIFAR-10 Federated Learning Experiment ==========
Random Seed:       42
Dataset Type:      IID
Quantization Type: 2
Quantization Rate: 4
Apply Quantization: 1
============================================================
```

### Load and Inspect a Result
```matlab
load('./results/2D_Lattice_4_iid_seed_1.mat');
plot(error_result);
xlabel('Iteration');
ylabel('Accuracy');
% error_result is 40×1 vector of accuracies
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Results not saving" | Check `./results/` exists, or set `'save_results', false` |
| "Out of memory error" | Reduce batch size in `trainingOptions` (~line 168) |
| "GetUnbalancedCIFAR not found" | Ensure `GetUnbalancedCIFAR.m` is in same directory |
| "Different results same seed" | Random GPU operations; use `gpuDevice('reset')` |

---

## Comparison: Old vs New

| Task | Old (CIFAR.m) | New |
|------|---|---|
| Single experiment | Hard-coded values | `run_cifar_experiment()` |
| Change seed | Edit file | `'seed', 42` parameter |
| Multiple seeds | Manual loop | Built-in auto-save |
| IID/non-IID toggle | Comment/uncomment | `'dataset_type'` parameter |
| Change quantization | Edit file | `'quant_type'` parameter |
| Batch testing | Write script | `run_batch_experiments()` |
| Results organization | Manual naming | Auto-generates standardized filenames |

---

## Next Steps

1. **Run a test:** 
   ```matlab
   run_cifar_experiment('seed', 1, 'quant_type', 2, 'proposed', 1)
   ```

2. **Collect error bars:**
   ```matlab
   for seed = 1:5
       run_cifar_experiment('seed', seed, 'quant_type', 2, 'quant_rate', 4, 'proposed', 1);
   end
   ```

3. **Full batch testing:**
   ```matlab
   run_batch_experiments()
   ```

4. **Check results:**
   ```matlab
   ls ./results/
   ```

---

**Questions?** Check `investigate.md` for random seed details or `QUICK_START.m` for copy-paste examples.
