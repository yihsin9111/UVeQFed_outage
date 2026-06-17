% ========================================================================
% Run CIFAR Experiments with Different Seeds/Settings and Save Results
% ========================================================================
% 
% USAGE: Edit the configuration below and run this script
%
% Results are auto-saved to ./results/ with naming:
%   <scheme>_<rate>_<dataset>_seed_<seed>.mat
%
% ========================================================================

%% Configuration - specify multiple values for batch testing

seeds = [1];                          % Seeds for error bars

% Dataset configurations (cell array)
dataset_configs = {
    'iid',      [];             % IID (ratio ignored)
    'non-iid',  0.25;            % Non-IID with ratio=0.25
};

% Quantization configurations (each row: [type, rate, proposed])
quant_configs = [
    2,  32,  0;                  % 2D Lattice, rate=32, no quant
    2,  2,  1;                  % 2D Lattice, rate=2, with quant
    4,  2,  1;                  % QSGD, rate=2, with quant
];

%% Run experiments in batch
num_datasets = size(dataset_configs, 1);
num_quants = size(quant_configs, 1);
total_exps = length(seeds) * num_datasets * num_quants;

fprintf('\n');
fprintf('========== BATCH EXPERIMENT RUNNER ==========\n');
fprintf('Total experiments: %d\n', total_exps);
fprintf('  Seeds: %d\n', length(seeds));
fprintf('  Dataset configs: %d\n', num_datasets);
fprintf('  Quantization configs: %d\n', num_quants);
fprintf('============================================\n\n');

% Create results directory
if ~exist('./results', 'dir')
    mkdir('./results');
end

%% Run all combinations
exp_count = 0;
all_results = {};

for d = 1:num_datasets
    dataset_type = dataset_configs{d, 1};
    ratio = dataset_configs{d, 2};
    
    for q = 1:num_quants
        quant_type = quant_configs(q, 1);
        quant_rate = quant_configs(q, 2);
        proposed = quant_configs(q, 3);
        
        fprintf('\n>>> Config %d/%d: dataset=%s', (d-1)*num_quants + q, num_datasets*num_quants, dataset_type);
        if ~isempty(ratio)
            fprintf(' (ratio=%.2f)', ratio);
        end
        fprintf(', quant_type=%d, rate=%.1f, proposed=%d\n', quant_type, quant_rate, proposed);
        
        errors_all = [];
        
        for seed = seeds
            exp_count = exp_count + 1;
            fprintf('  [%d/%d] seed=%d ... ', exp_count, total_exps, seed);
            
            % Run experiment
            if strcmpi(dataset_type, 'iid')
                error_curve = CIFAR('seed', seed, 'dataset_type', dataset_type, ...
                    'quant_type', quant_type, 'quant_rate', quant_rate, 'proposed', proposed);
            else
                error_curve = CIFAR('seed', seed, 'dataset_type', dataset_type, 'ratio', ratio, ...
                    'quant_type', quant_type, 'quant_rate', quant_rate, 'proposed', proposed);
            end
            
            % Save result
            scheme_names = {'3D_Lattice', '2D_Lattice', 'Scalar', 'QSGD', 'Unitary_Rot', 'Subsampling'};
            scheme_name = scheme_names{quant_type};
            
            if strcmpi(dataset_type, 'iid')
                filename = sprintf('%s_%.1f_%s_seed_%d.mat', scheme_name, quant_rate, dataset_type, seed);
            else
                filename = sprintf('%s_%.1f_%s_ratio_%.2f_seed_%d.mat', scheme_name, quant_rate, dataset_type, ratio, seed);
            end
            
            filepath = fullfile('./results', filename);
            save(filepath, 'error_curve');
            fprintf('saved\n');
            
            errors_all = [errors_all, error_curve];
        end
        
        % Store result summary
        mean_acc = mean(errors_all, 2);
        std_acc = std(errors_all, [], 2);
        
        result = struct();
        result.dataset_type = dataset_type;
        result.ratio = ratio;
        result.quant_type = quant_type;
        result.quant_rate = quant_rate;
        result.proposed = proposed;
        result.mean_error = mean_acc;
        result.std_error = std_acc;
        result.all_errors = errors_all;
        
        all_results{length(all_results)+1} = result;
        
        fprintf('  Final Accuracy: %.4f ± %.4f\n', mean_acc(end), std_acc(end));
    end
end

%% Save summary
save('./results/summary.mat', 'all_results');
fprintf('\n??? Summary saved to ./results/summary.mat\n');

%% Plot comparisons
fprintf('\nGenerating plots...\n');

figure('Position', [100 100 1200 800]);

% Find and plot all IID results
iid_results = [];
for i = 1:length(all_results)
    if strcmpi(all_results{i}.dataset_type, 'iid')
        iid_results = [iid_results, i];
    end
end

if ~isempty(iid_results)
    subplot(1, 2, 1);
    hold on;
    iterations = 1:length(all_results{iid_results(1)}.mean_error);
    
    for i = iid_results
        r = all_results{i};
        label = sprintf('Type=%d, Rate=%.1f, Prop=%d', r.quant_type, r.quant_rate, r.proposed);
        errorbar(iterations, r.mean_error, r.std_error, 'o-', 'LineWidth', 2, 'DisplayName', label);
    end
    
    xlabel('Iteration', 'FontSize', 12);
    ylabel('Accuracy', 'FontSize', 12);
    title('IID Dataset Results', 'FontSize', 13, 'FontWeight', 'bold');
    legend('FontSize', 10);
    grid on;
    hold off;
end

% Find and plot all non-IID results
noniid_results = [];
for i = 1:length(all_results)
    if strcmpi(all_results{i}.dataset_type, 'non-iid')
        noniid_results = [noniid_results, i];
    end
end

if ~isempty(noniid_results)
    subplot(1, 2, 2);
    hold on;
    iterations = 1:length(all_results{noniid_results(1)}.mean_error);
    
    for i = noniid_results
        r = all_results{i};
        label = sprintf('Ratio=%.2f, Type=%d, Prop=%d', r.ratio, r.quant_type, r.proposed);
        errorbar(iterations, r.mean_error, r.std_error, 'o-', 'LineWidth', 2, 'DisplayName', label);
    end
    
    xlabel('Iteration', 'FontSize', 12);
    ylabel('Accuracy', 'FontSize', 12);
    title('Non-IID Dataset Results', 'FontSize', 13, 'FontWeight', 'bold');
    legend('FontSize', 10);
    grid on;
    hold off;
end

sgtitle('CIFAR-10 FL Experiments', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('\n========== COMPLETE ==========\n');
fprintf('Results in ./results/\n');
fprintf('Summary: ./results/summary.mat\n');
fprintf('==============================\n\n');
