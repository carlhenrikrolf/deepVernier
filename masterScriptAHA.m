%% Load net
if 1
disp('loading net');
net = load('../nets/imagenet-caffe-alex.mat'); % Load the network
net = vl_simplenn_tidy(net);                   % Update and fill in values
close all
end

%% Parameters
trainType = 'vernier';
nRuns = 20;
readoutLayers = [2, 6, 10, 12, 14, 17, 19, 20];
trainSize    = 10000;         % There are 2*trainSize training samples, because L and R
imSize       = [227,227];
nUncrowded   = 0;       % 1 = 3 squares ; 2 = 5 squares ; 3 = 7 squares

net.meta.trainOpts.learningRate = 0.001 ;
net.meta.trainOpts.numEpochs = 20 ;
net.meta.trainOpts.batchSize = 1 ;

if nUncrowded
    nExperiments = 1 + 1 + length(nUncrowded); % vernier + crowded + uncrowded
else
    nExperiments = 2;
end
testSize     = round(0.15*trainSize);
% seed         = 1995; rng(seed);
stimSizes = {'small' 'medium' 'large'};
allAccuracies = zeros(length(stimSizes), nRuns, length(readoutLayers), nExperiments+1);
allMSEs = zeros(length(stimSizes), nRuns, length(readoutLayers), nExperiments+1);

for currentStim = 1:length(stimSizes)
    currentSize = stimSizes{currentStim};
    if strcmp(currentSize,'small')
        D            = 1:5;       % various vernier offsets,   in pixels
        T            = 1:3;       % various vernier thickness, 
        L            = 3:7;       % various lengths (1 bar),   in pixels 
    elseif strcmp(currentSize,'medium')
        D            = 1:10;
        T            = 1:5;
        L            = 5:12;
    elseif strcmp(currentSize,'large')
        D            = 1:15;
        T            = 1:8;
        L            = 5:round(imSize(1)/4);
    else
        error('You must define stimSizes')
    end

    %% Create sets
    disp('creating stimuli');
    if strcmp(trainType,'vernier')
        [trainSet, vernierTestSet, trainAnswers, vernierTestAnswers] = makeTrainingAndTestingSampleSets(       trainSize, testSize, imSize, D, T, L);
        [~,        crowdedTestSet, ~,            crowdedTestAnswers] = makeCrowdedTrainingAndTestingSampleSets(trainSize, testSize, imSize, D, T, L);
    else
        [~,        vernierTestSet, ~,            vernierTestAnswers] = makeTrainingAndTestingSampleSets(       trainSize, testSize, imSize, D, T, L);
        [trainSet, crowdedTestSet, trainAnswers, crowdedTestAnswers] = makeCrowdedTrainingAndTestingSampleSets(trainSize, testSize, imSize, D, T, L);
    end
    uncrowdedTestSets = cell(1,length(nUncrowded));
    uncrowdedTestAnswers = cell(1,length(nUncrowded));

    if nUncrowded
        for i = 1:length(nUncrowded)
            [~, uncrowdedTestSets{1,i}, ~, uncrowdedTestAnswers{1,i}] = makeUncrowdedTrainingAndTestingSampleSets(trainSize, testSize, imSize, D, T, L, nUncrowded(i));
        end
    end
    % figure()
    % for i = 1:testSize
    %     imagesc(crowdedTestSet(:,:,i))
    %     drawnow
    % end
    
    trainImdb = makeImdb(trainSet,trainAnswers);
    vernierImdb = makeImdb(vernierTestSet(:,:,1:testSize),vernierTestAnswers(1:testSize));
    crowdedImdb = makeImdb(crowdedTestSet(:,:,1:testSize),crowdedTestAnswers(1:testSize));

    for run = 1:nRuns
        %% Build Training sets filtered through DNN, train and test for all conditions
        accuracies = zeros(length(readoutLayers),nExperiments+1);
        MSEs = zeros(length(readoutLayers),nExperiments+1);
        disp('starting hyperloop')
        N = 0;
        for currentLayer = readoutLayers
            
            N = N+1;
            
            %% Add softmax
            net.layers{1,N:end} = [];
            net.layers{1,N} = struct('type','softmaxloss');
            
            %% Train
            [net, info] = cnn_train(net,trainImdb,getBatch,...
                net.meta.trainOpts,...
                'backDropDepth',1,...
                'val',find(imdb.images.set == 3));
            
            %% Get data 
            [accuracies(N,1), MSEs(N,1)] = getResult(net,trainImdb.images.data(:,:,:,1:testSize),answers(1:testSize));

            %% Test the classifier
            for k = 1:nExperiments
                if k == 1
                    imdb = vernierImdb;
                elseif k == 2
                    imdb = crowdedImdb;
                else
%                     testSet       = uncrowdedTestSets{   1,k-2};
%                     testAnswers   = uncrowdedTestAnswers{1,k-2};
                end
                [accuracies(N,k+1), MSEs(N,k+1)] = getResult(net,imdb.images.data,imdb.images.labels);
            end
            disp(['currentLayerFinished = ', num2str(currentLayer)])
        end
        allAccuracies(currentStim, run, :, :) = accuracies;
        allMSEs(currentStim, run, :, :) = MSEs;
    end
end
%% Plot and save data

                                         
                                         cd data
                                         save('allAccuracies','allAccuracies')
                                         save('allMSEs','allMSEs')
                                         cd ..
                                         
                                         mean_accuracies = mean(allAccuracies, 2);
                                         std_accuracies = std(allAccuracies, 0, 2);

for stimSize = 1:length(stimSizes)
    
    figure(stimSize)
    hold on
    plotAccuracies(squeeze(mean_accuracies(stimSize, 1, :, :)), squeeze(std_accuracies(stimSize, 1, :, :)))

end


disp('I''m glad that''s done.')


