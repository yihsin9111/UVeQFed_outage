% ============================================================================
% CIFAR-10 Federated Learning: Batch Experiment Runner
% ============================================================================
% This script demonstrates how to run multiple experiments with different
% random seeds, datasets, and quantization settings for error bar plotting.
%
% Usage: Simply adjust the configuration below and run this script
% ============================================================================

clear all; close all;

%% ========== EXPERIMENT CONFIGURATION ==========

% Number of experiments per configuration (for error bars)
num_seeds = 5;
seeds = 1:num_seeds;

% Dataset configurations to test
% Format: {dataset_type, ratio (only for non-iid)}
dataset_configs = {
    'iid', [];           % IID dataset
    'non-iid', 0.3;      % Non-IID with ratio=0.3
    'non-iid', 0.5;      % Non-IID with ratio=0.5
};

% Quantization configurations to test
% Format: {quant_type, quant_rate, proposed}
quant_configs = {
    2, 4, 0;             % 2D Lattice, rate=4, no quantization
    2, 4, 1;             % 2D Lattice, rate=4, WITH quantization
    3, 4, 1;             % Scalar, rate=4, WITH quantization
    4, 4, 1;             % QSGD, rate=4, WITH quantization
};

%% ========== RUN EXPERIMENTS ==========

num_datasets = size(dataset_configs, 1);
num_quants = size(quant_configs, 1);
total_experiments = num_seeds * num_datasets * num_quants;

fprintf('\n');
fprintf('===============================================\n');
fprintf('  CIFAR-10 FL: BATCH EXPERIMENT RUNNER\n');
fprintf('===============================================\n');
fprintf('Total Experiments: %d\n', total_experiments);
fprintf('  - Seeds: %d\n', num_seeds);
fprintf('  - Dataset Configs: %d\n', num_datasets);
fprintf('  - Quantization Configs: %d\n', num_quants);
fprintf('===============================================\n\n');

% Store all results
all_results = {};
result_idx = 1;

% Loop over all configurations
for d = 1:num_datasets
    dataset_type = dataset_configs{d, 1};
    ratio = dataset_configs{d, 2};
    
    for q = 1:num_quants
        quant_type = quant_configs{q, 1};
        quant_rate = quant_configs{q, 2};
        proposed = quant_configs{q, 3};
        
        fprintf('\n>>> Configuration %d/%d\n', d + (q-1)*num_datasets, num_datasets*num_quants);
        fprintf('    Dataset: %s', upper(dataset_type));
        if strcmpi(dataset_type, 'non-iid')
            fprintf(' (ratio=%.2f)', ratio);
        end
        fprintf('\n');
        fprintf('    Quantization: Type=%d, Rate=%.2f, Applied=%d\n', ...
            quant_type, quant_rate, proposed);
        fprintf('    Running %d seeds...\n', num_seeds);
        
        % Run experiments for this configuration with different seeds
        errors_matrix = [];
        
        for s = 1:num_seeds
            seed = seeds(s);
            
            % Run experiment
            if strcmpi(dataset_type, 'iid')
                error_curve = run_cifar_experiment( ...
                    'seed', seed, ...
                    'dataset_type', dataset_type, ...
                    'quant_type', quant_type, ...
                    'quant_rate', quant_rate, ...
                    'proposed', proposed, ...
                    'save_results', true);
            else
                error_curve = run_cifar_experiment( ...
                    'seed', seed, ...
                    'dataset_type', dataset_type, ...
                    'ratio', ratio, ...
                    'quant_type', quant_type, ...
                    'quant_rate', quant_rate, ...
                    'proposed', proposed, ...
                    'save_results', true);
            end
            
            errors_matrix = [errors_matrix, error_curve];
        end
        
        % Calculate statistics
        mean_error = mean(errors_matrix, 2);
        std_error = std(errors_matrix, [], 2);
        
        % Store results
        all_results{result_idx}.dataset_type = dataset_type;
        all_results{result_idx}.ratio = ratio;
        all_results{result_idx}.quant_type = quant_type;
        all_results{result_idx}.quant_rate = quant_rate;
        all_results{result_idx}.proposed = proposed;
        all_results{result_idx}.mean_error = mean_error;
        all_results{result_idx}.std_error = std_error;
        all_results{result_idx}.errors_all = errors_matrix;
        
        result_idx = result_idx + 1;
        
        fprintf('    ✓ Completed with mean final accuracy = %.4f ± %.4f\n', ...
            mean_error(end), std_error(end));
    end
end

%% ========== PLOTTING RESULTS ==========

fprintf('\n\n>>> Generating plots...\n\n');

% Plot 1: IID vs Non-IID comparison (baseline, no quantization)
figure('Position', [100 100 1000 600]);

% Find baseline configs
iid_baseline_idx = [];
noniid_baseline_idx = [];

for r = 1:length(all_results)
    if strcmpi(all_results{r}.dataset_type, 'iid') && all_results{r}.proposed == 0
        iid_baseline_idx = r;
    end
    if strcmpi(all_results{r}.dataset_type, 'non-iid') && all_results{r}.proposed == 0
        noniid_baseline_idx = r;
    end
end

if ~isempty(iid_baseline_idx) && ~isempty(noniid_baseline_idx)
    subplot(1, 2, 1);
    iterations = 1:length(all_results{iid_baseline_idx}.mean_error);
    
    hold on;
    errorbar(iterations, all_results{iid_baseline_idx}.mean_error, ...
        all_results{iid_baseline_idx}.std_error, 'o-', 'LineWidth', 2, 'MarkerSize', 6);
    errorbar(iterations, all_results{noniid_baseline_idx}.mean_error, ...
        all_results{noniid_baseline_idx}.std_error, 's-', 'LineWidth', 2, 'MarkerSize', 6);
    
    xlabel('Iteration', 'FontSize', 12);
    ylabel('Accuracy', 'FontSize', 12);
    title('IID vs Non-IID Baseline (No Quantization)', 'FontSize', 13, 'FontWeight', 'bold');
    legend('IID', sprintf('Non-IID (ratio=%.2f)', all_results{noniid_baseline_idx}.ratio), 'FontSize', 11);
    grid on;
    hold off;
end

% Plot 2: Quantization effect (best configuration)
subplot(1, 2, 2);

% Find quantization comparison (same dataset, with/without quant)
with_quant_idx = [];
for r = 1:length(all_results)
    if all_results{r}.proposed == 1 && strcmpi(all_results{r}.dataset_type, 'iid')
        with_quant_idx = [with_quant_idx, r];
    end
end

hold on;
if ~isempty(iid_baseline_idx)
    iterations = 1:length(all_results{iid_baseline_idx}.mean_error);
    errorbar(iterations, all_results{iid_baseline_idx}.mean_error, ...
        all_results{iid_baseline_idx}.std_error, 'o-', 'LineWidth', 2, 'MarkerSize', 6, 'Label', 'No Quantization');
end

quant_names = {'3D Lattice', '2D Lattice', 'Scalar', 'QSGD', 'Unitary Rot', 'Subsampling'};
colors = lines(length(with_quant_idx));

for i = 1:length(with_quant_idx)
    r = with_quant_idx(i);
    errorbar(iterations, all_results{r}.mean_error, all_results{r}.std_error, ...
        '-', 'LineWidth', 2, 'MarkerSize', 6, 'Color', colors(i, :), ...
        'Label', sprintf('%s (rate=%.1f)', quant_names{all_results{r}.quant_type}, all_results{r}.quant_rate));
end

xlabel('Iteration', 'FontSize', 12);
ylabel('Accuracy', 'FontSize', 12);
title('Effect of Quantization Schemes (IID Dataset)', 'FontSize', 13, 'FontWeight', 'bold');
legend('FontSize', 10, 'Location', 'best');
grid on;
hold off;

sgtitle('CIFAR-10 Federated Learning: Experiment Results', 'FontSize', 14, 'FontWeight', 'bold');

%% ========== SAVE SUMMARY ==========

% Save summary
summary_file = './results/experiment_summary.mat';
save(summary_file, 'all_results', 'dataset_configs', 'quant_configs', 'seeds');
fprintf('\n✓ Summary saved to: %s\n', summary_file);

fprintf('\n===============================================\n');
fprintf('  All experiments completed!\n');
fprintf('  Results saved to ./results/\n');
fprintf('===============================================\n\n');
