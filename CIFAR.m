
function error_result = CIFAR(varargin)
% CIFAR-10 Federated Learning Experiment
% Syntax: error_result = CIFAR('seed', 42, 'quant_type', 2, 'quant_rate', 4, 'proposed', 1)
% Parameters: 'seed', 'dataset_type' ('iid'/'non-iid'), 'ratio', 'quant_type', 'quant_rate', 'proposed'
clearvars -except varargin
% Parse inputs
p = inputParser;
addParameter(p, 'seed', 1, @isnumeric);
addParameter(p, 'dataset_type', 'iid', @ischar);
addParameter(p, 'ratio', 0.5, @isnumeric);
addParameter(p, 'quant_type', 2, @isnumeric);
addParameter(p, 'quant_rate', 4, @isnumeric);
addParameter(p, 'proposed', 0, @isnumeric);
parse(p, varargin{:});

seed_val = p.Results.seed;
dataset_type = p.Results.dataset_type;
ratio = p.Results.ratio;
proposed = p.Results.proposed;

% Set random seed
rng(seed_val);

datanumber=5000;  %% the number of data samples of each user

%Run DownloadCIFAR10 function to download CIFAR-10 dataset
%Run
% %% Prepare the CIFAR-10 dataset
% if ~exist('cifar10Train','dir')
%     disp('Saving the Images in folders. This might take some time...');    
%     saveCIFAR10AsFolderOfImages('cifar-10-batches-mat', pwd, true);
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%% data processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
categories = {'Deer','Dog','Frog','Cat','Bird','Automobile','Horse','Ship','Truck','Airplane'};

rootFolder = 'cifar10Test';
imds_test = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');


 categories = {'Deer','Dog','Frog','Cat','Bird','Automobile','Horse','Ship','Truck','Airplane'};

rootFolder = 'cifar10Train';
imds = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');
 
%%%%%%%%%%%%%%%%%%%%% IID dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
if strcmpi(dataset_type, 'iid')
    fprintf('  iid dataset\n');
    [imds1,imds2,imds3,imds4,imds5,imds6,imds7,imds8,imds9,imds10] = splitEachLabel(imds, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1);
else
    fprintf('  non-iid dataset with ratio %d\n', ratio);
    [imds1,imds2,imds3,imds4,imds5,imds6,imds7,imds8,imds9,imds10] = GetUnbalancedCIFAR(rootFolder, ratio);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% imds1 is the dataset of user 1. 


numberofneuron=50; % Number of neurons that consists of local FL model of each user
averagenumber=1;  % Average number of runing simulations. 
iteration=40;     % Total number of global FL iterations.
learningspeed=0.005; % Learning speed of each user
q = 0;             % Per-device outage probability (Bernoulli, i.i.d. across k and t)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%% coding setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
varSize = 32; 
v_fQRate = [1, 2];
v_nQuantizaers   = [...          % Curves
    0 ...                   % Dithered 3-D lattice quantization 
    1 ...                   % Dithered 2-D lattice quantization    
    1 ...                   % Dithered scalar quantization      
    1 ...                   % QSGD 
    1 ...                   % Uniform quantization with random unitary rotation    
    1 ...                   % Subsampling with 3 bits quantizers
    ];

global gm_fGenMat2D;
global gm_fLattice2D;
% Clear lattices
gm_fGenMat2D = [];
gm_fLattice2D = [];
% Do full search over the lattice
stSettings.OptSearch = 1;

stSettings.type = p.Results.quant_type;
stSettings.scale=1;
s_fRate = p.Results.quant_rate;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

localiterations=17;  % Number of local updates at each iteration.


finalerror=[];
averageerror=[];
kk=0;



%%%%%%%%%%%%%%%%%%%%%%%% Matrix size of local FL model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1length=5*5*3*32;

w2length=5*5*32*32;

w3length=5*5*32*64;

w4length=64*576;

w5length=10*64;

b1length=32;

b2length=32;

b3length=64;

b4length=64;

b5length=10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

error=zeros(iteration,1);

for userno=10:3:10    % Number of users.
    kk=kk+1;
    usernumber=userno; 
    
    
for average=1:1:averagenumber

    
    
wupdate=zeros(iteration,usernumber);   % local model for each user

% --- Outage tracking ---
outage_log         = false(iteration, usernumber); % full K×1 mask each iteration
active_devices_log = zeros(iteration, 1);          % sum(outage_mask) per iteration
retry_log          = zeros(iteration, 1);          % 1 if all-outage retry occurred
retry_count        = 0;                            % cumulative retry events

%%%%%%%%%%%%% local model of each user%%%%%%%%%%%%%%%%%%%%%%%  
w1=[];
w2=[];
w3=[];
w4=[];
w5=[];
b1=[];
b2=[];
b3=[];
b4=[];
b5=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

wnew=zeros(5,5,3,32,usernumber);
lwnew=zeros(5,5,32,32,usernumber);
bnew=zeros(5,5,32,64,usernumber);
obnew=zeros(64,576,usernumber);
fwnew=zeros(10,64,usernumber);



%%%%%%%%%%%%% gradient of local FL models %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
deviationw=[];
deviationlw=[];
deviationb=[];
deviationob=[];
deviationofw=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%Building local FL model of each user  %%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




for i=1:1:iteration
    
%%%%%%%%%%%%%%%%%%%%%%%Setting of local FL model %%%%%%%%%%%%%%%%%%%%%%%%%%   
    if i==16
        option = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 60, ...
    'Verbose', false);
    
     elseif i==25
        option = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.002, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 60, ...
    'Verbose', false);
     elseif i==33
        option = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 60, ...
    'Verbose', false);
     elseif i==39
        option = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 60, ...
    'Verbose', false);
         elseif i==42
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Outage mask (Algorithm 3, Steps 14-16) ---
outage_mask = (rand(usernumber, 1) > q);       % 1_k^t ~ Bernoulli(1-q), outage probability = q
if sum(outage_mask) == 0                       % all-outage: retry until ≥1 active
    retry_count  = retry_count + 1;
    retry_log(i) = 1;
    while sum(outage_mask) == 0
        outage_mask = (rand(usernumber, 1) > q);
    end
end
outage_log(i, :)      = outage_mask';           % for record : column to row vector
active_devices_log(i) = sum(outage_mask);       % for record : total active devices

for user=1:1:usernumber
       
    
           clear netvaluable;
    Winstr1=strcat('net',int2str(user));     
     midstr=strcat('imds',int2str(user)); 
     
    eval(['imdss','=',midstr,';']);
    
if i > 1
   % Let global FL model to be the local FL model of each user, which is
   % equal to that the BS transmits the global FL model to the users  

      layer(2).Weights=globalw1;

    layer(5).Weights=globalw2;

     layer(8).Weights=globalw3;
     layer(11).Weights=globalw4;
    layer(13).Weights=globalw5;   
     
         layer(2).Bias=globalb1;

    layer(5).Bias=globalb2;

     layer(8).Bias=globalb3;
     layer(11).Bias=globalb4;
    layer(13).Bias=globalb5;   
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
       
      




[netvaluable, info] = trainNetwork(imdss, layer, option); % Train local FL model.


%%%%%%%%%%%%%%%%%%%calculate identification accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
b_netDiverged = false;
for learnLayer = [2 5 8 11 13]
    if any(~isfinite(netvaluable.Layers(learnLayer).Weights(:))) || ...
       any(~isfinite(netvaluable.Layers(learnLayer).Bias(:)))
        b_netDiverged = true;
        break;
    end
end

if b_netDiverged
    warning('UVeQFed:nonFiniteLocalModel', ...
        'Iteration %d, user %d: local model diverged (non-finite weights) — skipping accuracy this round.', i, user);
else
 labels = classify(netvaluable, imds_test);

% This could take a while if you are not using a GPU
confMat = confusionmat(imds_test.Labels, labels);
confMat = confMat./sum(confMat,2);
error(i,1)=mean(diag(confMat))+error(i,1); % Here, error is identification accuracy.
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%% global model for each user, which consists of 4 matrices  

if i==1    
    globalw1=zeros(5,5,3,32);
globalw2=zeros(5,5,32,32);
globalw3=zeros(5,5,32,64);
globalw4=zeros(64,576);
globalw5=zeros(10,64);

    globalb1=zeros(1,1,32);
globalb2=zeros(1,1,32);
globalb3=zeros(1,1,64);
globalb4=zeros(64,1);
globalb5=zeros(10,1);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Record trained local FL model.

w1(:,:,:,:,user)=netvaluable.Layers(2).Weights;

w2(:,:,:,:,user)=netvaluable.Layers(5).Weights;

     w3(:,:,:,:,user)=netvaluable.Layers(8).Weights;
    w4(:,:,user)=netvaluable.Layers(11).Weights;
w5(:,:,user)=netvaluable.Layers(13).Weights;
     
     
b1(:,:,:,user)=netvaluable.Layers(2).Bias;

b2(:,:,:,user)=netvaluable.Layers(5).Bias;

   b3(:,:,:,user)=netvaluable.Layers(8).Bias;
   b4(:,:,user)=netvaluable.Layers(11).Bias;
   b5(:,:,user)=netvaluable.Layers(13).Bias;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     
if proposed==1
    
    
%%%%%%%%%%%%% Calculate the gradient of local FL model of each user%%%%%%%        
if i==1    
     
deviationw1= w1(:,:,:,:,user);
deviationw2=w2(:,:,:,:,user);
deviationw3= w3(:,:,:,:,user);
deviationw4=w4(:,:,user);
deviationw5=w5(:,:,user);

deviationb1=b1(:,:,:,user);
deviationb2=b2(:,:,:,user);
deviationb3=b3(:,:,:,user);
deviationb4=b4(:,:,user);
deviationb5= b5(:,:,user);

else
    
    
deviationw1= w1(:,:,:,:,user)-globalw1;
deviationw2=w2(:,:,:,:,user)-globalw2;
deviationw3= w3(:,:,:,:,user)-globalw3;
deviationw4=w4(:,:,user)-globalw4;
deviationw5=w5(:,:,user)-globalw5;

deviationb1=b1(:,:,:,user)-globalb1;
deviationb2=b2(:,:,:,user)-globalb2;
deviationb3=b3(:,:,:,user)-globalb3;
deviationb4=b4(:,:,user)-globalb4;
deviationb5= b5(:,:,user)-globalb5;    
        
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%% reshape the gradient of local FL model of each user%%%%%%%        

w1vector=reshape(deviationw1,[w1length,1]);

w2vector=reshape(deviationw2,[w2length,1]);

w3vector=reshape(deviationw3,[w3length,1]);

w4vector=reshape(deviationw4,[w4length,1]);

w5vector=reshape(deviationw5,[w5length,1]);   


b1vector=reshape(deviationb1,[b1length,1]);

b2vector=reshape(deviationb2,[b2length,1]);

b3vector=reshape(deviationb3,[b3length,1]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





    m_fH1 = [w1vector;w2vector;w3vector;w4vector;w5vector;...
             b1vector;b2vector;b3vector;deviationb4;deviationb5];

   v_bNonFinite = ~isfinite(m_fH1); % catches Inf as well as NaN
   if any(v_bNonFinite)
       warning('UVeQFed:nonFiniteUpdate', ...
           'Iteration %d, user %d: %d non-finite entries in local update — treating as no-update.', ...
           i, user, sum(v_bNonFinite));
       m_fH1(v_bNonFinite) = 0;
   end
   [m_fHhat1, ~] = m_fQuantizeData(m_fH1, s_fRate, stSettings); % coding and decoding
 
   bstart=w1length+w2length+w3length+w4length+w5length;
   
 %%%%%%%%%%%%%%%% reshape the gradient of the loss function after coding %%%%%%%%%%%%  
 deviationw1=reshape(m_fHhat1(1:w1length),[5,5,3,32]);
  deviationw2=reshape(m_fHhat1(w1length+1:w1length+w2length),[5,5,32,32]);
  deviationw3=reshape(m_fHhat1(w1length+w2length+1:w1length+w2length+w3length),[5,5,32,64]);
deviationw4=reshape(m_fHhat1(w1length+w2length+w3length+1:w1length+w2length+w3length+w4length),[64,576]);
deviationw5=reshape(m_fHhat1(w1length+w2length+w3length+w4length+1:bstart),[10,64]);

 deviationb1(1,1,:)=reshape(m_fHhat1(bstart+1:bstart+b1length),[1,1,32]);
  deviationb2(1,1,:)=reshape(m_fHhat1(bstart+b1length+1:bstart+b1length+b2length),[1,1,32]);
  deviationb3(1,1,:)=reshape(m_fHhat1(bstart+b1length+b2length+1:bstart+b1length+b2length+b3length),[1,1,64]);
deviationb4(:,1)=m_fHhat1(bstart+b1length+b2length+b3length+1:bstart+b1length+b2length+b3length+b4length);
deviationb5(:,1)=m_fHhat1(bstart+b1length+b2length+b3length+b4length+1:bstart+b1length+b2length+b3length+b4length+b5length);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Zero out reconstructed gradient for outage devices (PS receives nothing) ---
if ~outage_mask(user)
    deviationw1 = zeros(size(deviationw1));
    deviationw2 = zeros(size(deviationw2));
    deviationw3 = zeros(size(deviationw3));
    deviationw4 = zeros(size(deviationw4));
    deviationw5 = zeros(size(deviationw5));
    deviationb1 = zeros(size(deviationb1));
    deviationb2 = zeros(size(deviationb2));
    deviationb3 = zeros(size(deviationb3));
    deviationb4 = zeros(size(deviationb4));
    deviationb5 = zeros(size(deviationb5));
end

 %%%%%%%%%%%%%%%% calculate the local FL model of each user after coding %%%%%%%%%%%%

    if i==1
   
   w1(:,:,:,:,user)=deviationw1;
w2(:,:,:,:,user)=deviationw2;
 w3(:,:,:,:,user)=deviationw3;
w4(:,:,user)=deviationw4;
w5(:,:,user)=deviationw5;

b1(:,:,:,user)=deviationb1;
b2(:,:,:,user)=deviationb2;
b3(:,:,:,user)=deviationb3;
b4(:,:,user)=deviationb4;
b5(:,:,user)=deviationb5;
              
    else       
      w1(:,:,:,:,user)=deviationw1+globalw1;
w2(:,:,:,:,user)=deviationw2+globalw2;
 w3(:,:,:,:,user)=deviationw3+globalw3;
w4(:,:,user)=deviationw4+globalw4;
w5(:,:,user)=deviationw5+globalw5;

b1(:,:,:,user)=deviationb1+globalb1;
b2(:,:,:,user)=deviationb2+globalb2;
b3(:,:,:,user)=deviationb3+globalb3;
b4(:,:,user)=deviationb4+globalb4;
b5(:,:,user)=deviationb5+globalb5;     
        
    end
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

end



 %%%%%%%%%%%%%%%% masked global model update (outage-aware, Algorithm 3) %%%%
% Use index selection (not mask multiplication) to preserve weight dtype.
% mask multiplication would promote single weights to double and corrupt
% the layer assignments in the next iteration.
active_idx   = find(outage_mask);   % indices of non-outage devices
active_count = numel(active_idx);   % guaranteed >= 1 by the retry loop

globalw1 = (1/active_count) * sum(w1(:,:,:,:, active_idx), 5);
globalw2 = (1/active_count) * sum(w2(:,:,:,:, active_idx), 5);
globalw3 = (1/active_count) * sum(w3(:,:,:,:, active_idx), 5);
globalw4 = (1/active_count) * sum(w4(:,:,      active_idx), 3);
globalw5 = (1/active_count) * sum(w5(:,:,      active_idx), 3);

globalb1 = (1/active_count) * sum(b1(:,:,:,   active_idx), 4);
globalb2 = (1/active_count) * sum(b2(:,:,:,   active_idx), 4);
globalb3 = (1/active_count) * sum(b3(:,:,:,   active_idx), 4);
globalb4 = (1/active_count) * sum(b4(:,:,      active_idx), 3);
globalb5 = (1/active_count) * sum(b5(:,:,      active_idx), 3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%% without coding and encoding 
% 
% wglobal(i,:)=1/usernumber*sum(w,2);  % global training model
% lwglobal(i,:)=1/usernumber*sum(lw,2);  % global training model
% bglobal(i,:)=1/usernumber*sum(b,2);
% obglobal(i,:)=1/usernumber*sum(ob,2);


%tmp_net = netvaluable.saveobj;

% netvaluable.Layers(2).Weights =globalw1;
% tmp_net.Layers(5).Weights =globalw2;
% tmp_net.Layers(8).Weights =globalw3;
% tmp_net.Layers(11).Weights =globalw4;
% tmp_net.Layers(13).Weights =globalw5;
% 
% tmp_net.Layers(2).Bias =globalb1;
% tmp_net.Layers(5).Bias =globalb2;
% tmp_net.Layers(8).Bias =globalb3;
% tmp_net.Layers(11).Bias =globalb4;
% tmp_net.Layers(13).Bias =globalb5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


error(i,1)=error(i,1)/10; %%%% calculate the final error
end

% --- Outage diagnostics ---
if retry_count > 5
    warning('UVeQFed:highRetries', ...
        'Retry count = %d is unexpectedly high (q=%.2f, K=%d). Verify outage model.', ...
        retry_count, q, usernumber);
end
fprintf('[Outage] q=%.2f | K=%d | retries=%d | mean active=%.2f\n', ...
    q, usernumber, retry_count, mean(active_devices_log));

figure('Name', sprintf('Outage Diagnostics (K=%d, q=%.2f)', usernumber, q));

subplot(2, 1, 1);
plot(1:iteration, active_devices_log, 'b-o', 'MarkerSize', 4, 'LineWidth', 1.2);
hold on;
yline(usernumber, 'r--', sprintf('K=%d', usernumber), 'LabelHorizontalAlignment', 'left');
xlabel('Iteration t');
ylabel('Active devices');
title('Number of Active Devices per Iteration');
ylim([0, usernumber + 1]);
grid on;

subplot(2, 1, 2);
stem(1:iteration, retry_log, 'r', 'filled', 'MarkerSize', 5);
xlabel('Iteration t');
ylabel('Retry occurred');
title(sprintf('All-Outage Retry Events per Iteration (total: %d)', retry_count));
ylim([-0.2, 1.5]);
grid on;

end
end
error_result = error; % return final error result of all users 
end