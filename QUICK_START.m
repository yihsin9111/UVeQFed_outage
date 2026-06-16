% ============================================================================
% QUICK START: Using the New Parameterized Experiment Framework
% ============================================================================
% Copy and run these examples in MATLAB command window or scripts
% ============================================================================

%% EXAMPLE 1: Single experiment with defaults
run_cifar_experiment();

%% EXAMPLE 2: Single experiment with custom seed
run_cifar_experiment('seed', 42);

%% EXAMPLE 3: Non-IID dataset with ratio=0.3
run_cifar_experiment('dataset_type', 'non-iid', 'ratio', 0.3, 'seed', 1);

%% EXAMPLE 4: Apply 2D Lattice quantization at rate 4
run_cifar_experiment(...
    'seed', 1, ...
    'quant_type', 2, ...           % 2D Lattice
    'quant_rate', 4, ...
    'proposed', 1 ...              % Apply quantization
);

%% EXAMPLE 5: Test QSGD with non-IID data
run_cifar_experiment(...
    'seed', 5, ...
    'dataset_type', 'non-iid', ...
    'ratio', 0.5, ...
    'quant_type', 4, ...           % QSGD
    'quant_rate', 3, ...
    'proposed', 1 ...
);

%% EXAMPLE 6: Run multiple seeds for error bars
errors_collection = [];
for seed = 1:5
    error_curve = run_cifar_experiment(...
        'seed', seed, ...
        'quant_type', 3, ...        % Scalar quantization
        'quant_rate', 2, ...
        'proposed', 1 ...
    );
    errors_collection = [errors_collection, error_curve];
end

% Calculate statistics
mean_accuracy = mean(errors_collection, 2);
std_accuracy = std(errors_collection, [], 2);

% Plot with error bars
figure;
iterations = 1:length(mean_accuracy);
errorbar(iterations, mean_accuracy, std_accuracy, 'o-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Accuracy');
title('Scalar Quantization (rate=2): Mean ± Std over 5 seeds');
grid on;

%% EXAMPLE 7: FULL BATCH RUN (all configurations)
% For comprehensive experiments with multiple seeds, datasets, and quant schemes:
run_batch_experiments();
% This will:
% - Run all combinations automatically
% - Save individual results to ./results/
% - Generate comparison plots
% - Create summary file

%% ============================================================================
% PARAMETER REFERENCE
% ============================================================================
%
% run_cifar_experiment(...
%     'seed',         (int)      random seed [1-2^32] (default: 1)
%     'dataset_type', (string)   'iid' or 'non-iid' (default: 'iid')
%     'ratio',        (double)   non-IID ratio [0-1] (default: 0.5, only for non-iid)
%     'quant_type',   (int)      1-6, see below (default: 2)
%     'quant_rate',   (double)   quantization bits/sample (default: 4)
%     'proposed',     (int)      0=no quantization, 1=apply (default: 0)
%     'save_results', (bool)     save to ./results/ (default: true)
% )
%
% Quantization Types:
%   1 = 3D Lattice quantization
%   2 = 2D Lattice quantization (recommended)
%   3 = Scalar quantization
%   4 = QSGD (Quantized SGD)
%   5 = Uniform with random unitary rotation
%   6 = Subsampling with 3-bit quantizers
%
% Output Naming Convention: ./results/<quant_type>_<rate>_<iid/non-iid>_seed_<seed>.mat
%   Examples:
%     2D_Lattice_4_iid_seed_1.mat
%     Scalar_2_non_iid_ratio_0.30_seed_3.mat
%     QSGD_4_iid_seed_10.mat
%
% ============================================================================

% TROUBLESHOOTING:
% Q: Results not saving?
%    A: Check if ./results/ directory exists, or run with 'save_results', false
%
% Q: Getting OOM errors?
%    A: Reduce 'datanumber' in run_cifar_experiment.m or use smaller batch sizes
%
% Q: Want to modify learning rates or iteration count?
%    A: Edit run_cifar_experiment.m lines ~150-180 (trainingOptions)
%       or iteration at line ~120

% ============================================================================
