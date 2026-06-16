function error_result = run_cifar_experiment(varargin)
% Run CIFAR-10 Federated Learning experiment with specified parameters
%
% Syntax:
%   error_result = run_cifar_experiment()                          % Use all defaults
%   error_result = run_cifar_experiment('seed', 42)                % Set specific parameters
%   error_result = run_cifar_experiment('seed', 42, 'quant_type', 2, 'quant_rate', 4)
%
% Parameters:
%   'seed' (int)            : Random seed (default: 1)
%   'dataset_type' (string) : 'iid' or 'non-iid' (default: 'iid')
%   'ratio' (double)        : Data ratio for non-IID (default: 0.5, only used if dataset_type='non-iid')
%   'quant_type' (int)      : Quantization type 1-6 (default: 2)
%                             1=3D Lattice, 2=2D Lattice, 3=Scalar, 4=QSGD, 5=Unitary Rot, 6=Subsampling
%   'quant_rate' (double)   : Quantization rate (default: 4)
%   'proposed' (int)        : Apply quantization? 0=no, 1=yes (default: 0)
%   'save_results' (bool)   : Save to ./results directory? (default: true)
%
% Output:
%   error_result : Accuracy curve across iterations (iterations x 1)
%
% Example:
%   err = run_cifar_experiment('seed', 5, 'quant_type', 2, 'quant_rate', 4, 'proposed', 1);
%   err = run_cifar_experiment('dataset_type', 'non-iid', 'ratio', 0.3, 'seed', 10);

    % Parse input arguments
    p = inputParser;
    addParameter(p, 'seed', 1, @isnumeric);
    addParameter(p, 'dataset_type', 'iid', @ischar);
    addParameter(p, 'ratio', 0.5, @isnumeric);
    addParameter(p, 'quant_type', 2, @isnumeric);
    addParameter(p, 'quant_rate', 4, @isnumeric);
    addParameter(p, 'proposed', 0, @isnumeric);
    addParameter(p, 'save_results', true, @islogical);
    
    parse(p, varargin{:});
    
    seed_val = p.Results.seed;
    dataset_type = p.Results.dataset_type;
    ratio = p.Results.ratio;
    quant_type = p.Results.quant_type;
    quant_rate = p.Results.quant_rate;
    proposed = p.Results.proposed;
    save_results = p.Results.save_results;
    
    % Set random seed
    rng(seed_val);
    
    % Display experiment info
    fprintf('\n========== CIFAR-10 Federated Learning Experiment ==========\n');
    fprintf('Random Seed:       %d\n', seed_val);
    fprintf('Dataset Type:      %s\n', upper(dataset_type));
    if strcmpi(dataset_type, 'non-iid')
        fprintf('Non-IID Ratio:     %.2f\n', ratio);
    end
    fprintf('Quantization Type: %d\n', quant_type);
    fprintf('Quantization Rate: %.2f\n', quant_rate);
    fprintf('Apply Quantization: %d\n', proposed);
    fprintf('============================================================\n\n');
    
    % ==================== Data Loading ====================
    categories = {'Deer','Dog','Frog','Cat','Bird','Automobile','Horse','Ship','Truck','Airplane'};
    
    % Test set (always IID)
    rootFolder_test = 'cifar10Test';
    imds_test = imageDatastore(fullfile(rootFolder_test, categories), ...
        'LabelSource', 'foldernames');
    
    % Training set
    rootFolder_train = 'cifar10Train';
    imds = imageDatastore(fullfile(rootFolder_train, categories), ...
        'LabelSource', 'foldernames');
    
    % Split data based on dataset type
    if strcmpi(dataset_type, 'iid')
        % IID split
        [imds1,imds2,imds3,imds4,imds5,imds6,imds7,imds8,imds9,imds10] = ...
            splitEachLabel(imds, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1);
    elseif strcmpi(dataset_type, 'non-iid')
        % Non-IID split
        [imds1,imds2,imds3,imds4,imds5,imds6,imds7,imds8,imds9,imds10] = ...
            GetUnbalancedCIFAR(rootFolder_train, ratio);
    else
        error('Invalid dataset_type. Use ''iid'' or ''non-iid''');
    end
    
    % ==================== Experiment Configuration ====================
    datanumber = 5000;          % Number of data samples per user
    numberofneuron = 50;        % Neurons in local FL model
    averagenumber = 1;          % Number of averaging runs
    iteration = 40;             % Total FL iterations
    localiterations = 17;       % Local updates per iteration
    
    % ==================== Quantization Settings ====================
    global gm_fGenMat2D gm_fLattice2D;
    gm_fGenMat2D = [];
    gm_fLattice2D = [];
    
    stSettings.OptSearch = 1;
    stSettings.type = quant_type;
    stSettings.scale = 1;
    s_fRate = quant_rate;
    
    % ==================== Model Architecture ====================
    varSize = 32;
    layer = [
        imageInputLayer([varSize varSize 3]);
        convolution2dLayer(5,varSize,'Padding',2,'BiasLearnRateFactor',2);
        maxPooling2dLayer(3,'Stride',2);
        reluLayer();
        convolution2dLayer(5,32,'Padding',2,'BiasLearnRateFactor',2);
        reluLayer();
        averagePooling2dLayer(3,'Stride',2);
        convolution2dLayer(5,64,'Padding',2,'BiasLearnRateFactor',2);
        reluLayer();
        averagePooling2dLayer(3,'Stride',2);
        fullyConnectedLayer(64,'BiasLearnRateFactor',2);
        reluLayer();
        fullyConnectedLayer(length(categories),'BiasLearnRateFactor',2);
        softmaxLayer()
        classificationLayer()];
    
    option = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.008, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 8, ...
        'L2Regularization', 0.004, ...
        'MaxEpochs', 1, ...
        'MiniBatchSize', 60, ...
        'Verbose', false);
    
    % ==================== Model Weight Sizes ====================
    w1length = 5*5*3*32;
    w2length = 5*5*32*32;
    w3length = 5*5*32*64;
    w4length = 64*576;
    w5length = 10*64;
    b1length = 32;
    b2length = 32;
    b3length = 64;
    b4length = 64;
    b5length = 10;
    
    % ==================== Initialize Error Array ====================
    error_result = zeros(iteration, 1);
    usernumber = 10;
    
    % ==================== Main FL Training Loop ====================
    for average = 1:averagenumber
        
        w1 = [];
        w2 = [];
        w3 = [];
        w4 = [];
        w5 = [];
        b1 = [];
        b2 = [];
        b3 = [];
        b4 = [];
        b5 = [];
        
        for i = 1:iteration
            
            % ========== Adjust learning rate ===========
            if i == 16
                option = trainingOptions('sgdm', ...
                    'InitialLearnRate', 0.005, ...
                    'LearnRateSchedule', 'piecewise', ...
                    'LearnRateDropFactor', 0.1, ...
                    'LearnRateDropPeriod', 8, ...
                    'L2Regularization', 0.004, ...
                    'MaxEpochs', 1, ...
                    'MiniBatchSize', 60, ...
                    'Verbose', false);
            elseif i == 25
                option = trainingOptions('sgdm', ...
                    'InitialLearnRate', 0.002, ...
                    'LearnRateSchedule', 'piecewise', ...
                    'LearnRateDropFactor', 0.1, ...
                    'LearnRateDropPeriod', 8, ...
                    'L2Regularization', 0.004, ...
                    'MaxEpochs', 1, ...
                    'MiniBatchSize', 60, ...
                    'Verbose', false);
            elseif i == 33
                option = trainingOptions('sgdm', ...
                    'InitialLearnRate', 0.0005, ...
                    'LearnRateSchedule', 'piecewise', ...
                    'LearnRateDropFactor', 0.1, ...
                    'LearnRateDropPeriod', 8, ...
                    'L2Regularization', 0.004, ...
                    'MaxEpochs', 1, ...
                    'MiniBatchSize', 60, ...
                    'Verbose', false);
            elseif i == 39
                option = trainingOptions('sgdm', ...
                    'InitialLearnRate', 0.0001, ...
                    'LearnRateSchedule', 'piecewise', ...
                    'LearnRateDropFactor', 0.1, ...
                    'LearnRateDropPeriod', 8, ...
                    'L2Regularization', 0.004, ...
                    'MaxEpochs', 1, ...
                    'MiniBatchSize', 60, ...
                    'Verbose', false);
            elseif i == 42
                option = trainingOptions('sgdm', ...
                    'InitialLearnRate', 0.00005, ...
                    'LearnRateSchedule', 'piecewise', ...
                    'LearnRateDropFactor', 0.1, ...
                    'LearnRateDropPeriod', 8, ...
                    'L2Regularization', 0.004, ...
                    'MaxEpochs', 1, ...
                    'MiniBatchSize', 60, ...
                    'Verbose', false);
            end
            
            % ========== Train each user's model ===========
            for user = 1:usernumber
                
                clear netvaluable;
                midstr = strcat('imds', int2str(user));
                eval(['imdss=', midstr, ';']);
                
                if i > 1
                    layer(2).Weights = globalw1;
                    layer(5).Weights = globalw2;
                    layer(8).Weights = globalw3;
                    layer(11).Weights = globalw4;
                    layer(13).Weights = globalw5;
                    
                    layer(2).Bias = globalb1;
                    layer(5).Bias = globalb2;
                    layer(8).Bias = globalb3;
                    layer(11).Bias = globalb4;
                    layer(13).Bias = globalb5;
                end
                
                % Train network
                [netvaluable, ~] = trainNetwork(imdss, layer, option);
                
                % Calculate accuracy
                labels = classify(netvaluable, imds_test);
                confMat = confusionmat(imds_test.Labels, labels);
                confMat = confMat ./ sum(confMat, 2);
                error_result(i, 1) = mean(diag(confMat)) + error_result(i, 1);
                
                % Record weights
                w1(:,:,:,:,user) = netvaluable.Layers(2).Weights;
                w2(:,:,:,:,user) = netvaluable.Layers(5).Weights;
                w3(:,:,:,:,user) = netvaluable.Layers(8).Weights;
                w4(:,:,user) = netvaluable.Layers(11).Weights;
                w5(:,:,user) = netvaluable.Layers(13).Weights;
                
                b1(:,:,:,user) = netvaluable.Layers(2).Bias;
                b2(:,:,:,user) = netvaluable.Layers(5).Bias;
                b3(:,:,:,user) = netvaluable.Layers(8).Bias;
                b4(:,:,user) = netvaluable.Layers(11).Bias;
                b5(:,:,user) = netvaluable.Layers(13).Bias;
                
                % ========== Apply quantization if proposed==1 ===========
                if proposed == 1
                    
                    if i == 1
                        deviationw1 = w1(:,:,:,:,user);
                        deviationw2 = w2(:,:,:,:,user);
                        deviationw3 = w3(:,:,:,:,user);
                        deviationw4 = w4(:,:,user);
                        deviationw5 = w5(:,:,user);
                        
                        deviationb1 = b1(:,:,:,user);
                        deviationb2 = b2(:,:,:,user);
                        deviationb3 = b3(:,:,:,user);
                        deviationb4 = b4(:,:,user);
                        deviationb5 = b5(:,:,user);
                    else
                        deviationw1 = w1(:,:,:,:,user) - globalw1;
                        deviationw2 = w2(:,:,:,:,user) - globalw2;
                        deviationw3 = w3(:,:,:,:,user) - globalw3;
                        deviationw4 = w4(:,:,user) - globalw4;
                        deviationw5 = w5(:,:,user) - globalw5;
                        
                        deviationb1 = b1(:,:,:,user) - globalb1;
                        deviationb2 = b2(:,:,:,user) - globalb2;
                        deviationb3 = b3(:,:,:,user) - globalb3;
                        deviationb4 = b4(:,:,user) - globalb4;
                        deviationb5 = b5(:,:,user) - globalb5;
                    end
                    
                    % Reshape gradients
                    w1vector = reshape(deviationw1, [w1length, 1]);
                    w2vector = reshape(deviationw2, [w2length, 1]);
                    w3vector = reshape(deviationw3, [w3length, 1]);
                    w4vector = reshape(deviationw4, [w4length, 1]);
                    w5vector = reshape(deviationw5, [w5length, 1]);
                    
                    b1vector = reshape(deviationb1, [b1length, 1]);
                    b2vector = reshape(deviationb2, [b2length, 1]);
                    b3vector = reshape(deviationb3, [b3length, 1]);
                    
                    m_fH1 = [w1vector; w2vector; w3vector; w4vector; w5vector; ...
                        b1vector; b2vector; b3vector; deviationb4; deviationb5];
                    
                    % Quantize
                    [m_fHhat1, ~] = m_fQuantizeData(m_fH1, s_fRate, stSettings);
                    
                    bstart = w1length + w2length + w3length + w4length + w5length;
                    
                    % Reshape after quantization
                    deviationw1 = reshape(m_fHhat1(1:w1length), [5,5,3,32]);
                    deviationw2 = reshape(m_fHhat1(w1length+1:w1length+w2length), [5,5,32,32]);
                    deviationw3 = reshape(m_fHhat1(w1length+w2length+1:w1length+w2length+w3length), [5,5,32,64]);
                    deviationw4 = reshape(m_fHhat1(w1length+w2length+w3length+1:w1length+w2length+w3length+w4length), [64,576]);
                    deviationw5 = reshape(m_fHhat1(w1length+w2length+w3length+w4length+1:bstart), [10,64]);
                    
                    deviationb1(1,1,:) = reshape(m_fHhat1(bstart+1:bstart+b1length), [1,1,32]);
                    deviationb2(1,1,:) = reshape(m_fHhat1(bstart+b1length+1:bstart+b1length+b2length), [1,1,32]);
                    deviationb3(1,1,:) = reshape(m_fHhat1(bstart+b1length+b2length+1:bstart+b1length+b2length+b3length), [1,1,64]);
                    deviationw4(:,1) = m_fHhat1(bstart+b1length+b2length+b3length+1:bstart+b1length+b2length+b3length+b4length);
                    deviationw5(:,1) = m_fHhat1(bstart+b1length+b2length+b3length+b4length+1:bstart+b1length+b2length+b3length+b4length+b5length);
                    
                    % Update models after quantization
                    if i == 1
                        w1(:,:,:,:,user) = deviationw1;
                        w2(:,:,:,:,user) = deviationw2;
                        w3(:,:,:,:,user) = deviationw3;
                        w4(:,:,user) = deviationw4;
                        w5(:,:,user) = deviationw5;
                        
                        b1(:,:,:,user) = deviationb1;
                        b2(:,:,:,user) = deviationb2;
                        b3(:,:,:,user) = deviationb3;
                        b4(:,:,user) = deviationb4;
                        b5(:,:,user) = deviationb5;
                    else
                        w1(:,:,:,:,user) = deviationw1 + globalw1;
                        w2(:,:,:,:,user) = deviationw2 + globalw2;
                        w3(:,:,:,:,user) = deviationw3 + globalw3;
                        w4(:,:,user) = deviationw4 + globalw4;
                        w5(:,:,user) = deviationw5 + globalw5;
                        
                        b1(:,:,:,user) = deviationb1 + globalb1;
                        b2(:,:,:,user) = deviationb2 + globalb2;
                        b3(:,:,:,user) = deviationb3 + globalb3;
                        b4(:,:,user) = deviationb4 + globalb4;
                        b5(:,:,user) = deviationb5 + globalb5;
                    end
                    
                end
                
            end
            
            % ========== Update global model ===========
            globalw1 = (1/usernumber) * sum(w1, 5);
            globalw2 = (1/usernumber) * sum(w2, 5);
            globalw3 = (1/usernumber) * sum(w3, 5);
            globalw4 = (1/usernumber) * sum(w4, 3);
            globalw5 = (1/usernumber) * sum(w5, 3);
            
            globalb1 = (1/usernumber) * sum(b1, 4);
            globalb2 = (1/usernumber) * sum(b2, 4);
            globalb3 = (1/usernumber) * sum(b3, 4);
            globalb4 = (1/usernumber) * sum(b4, 3);
            globalb5 = (1/usernumber) * sum(b5, 3);
            
            % Average error across users
            error_result(i, 1) = error_result(i, 1) / usernumber;
            
            fprintf('Iteration %2d / %d: Accuracy = %.4f\n', i, iteration, error_result(i, 1));
            
        end
        
    end
    
    % ==================== Save Results ====================
    if save_results
        results_dir = './results';
        if ~exist(results_dir, 'dir')
            mkdir(results_dir);
        end
        
        % Generate filename
        quantization_names = {'3D_Lattice', '2D_Lattice', 'Scalar', 'QSGD', 'Unitary_Rot', 'Subsampling'};
        if quant_type >= 1 && quant_type <= 6
            quant_name = quantization_names{quant_type};
        else
            quant_name = sprintf('Type%d', quant_type);
        end
        
        if strcmpi(dataset_type, 'iid')
            dataset_str = 'iid';
        else
            dataset_str = sprintf('non_iid_ratio_%.2f', ratio);
        end
        
        filename = sprintf('%s_%g_%s_seed_%d.mat', quant_name, quant_rate, dataset_str, seed_val);
        filepath = fullfile(results_dir, filename);
        
        save(filepath, 'error_result');
        fprintf('\nResults saved to: %s\n', filepath);
    end
    
end
