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

allClassifiers = cell(length(stimSizes),nRuns,length(readoutLayers));
allTrainResults = allClassifiers;

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
        [trainSet, vernierTestSet, trainAnswers, vernierTestAnswers] = makeTrainingAndTestingSampleSets(       trainSize, imSize, D, T, L);
        [~,        crowdedTestSet, ~,            crowdedTestAnswers] = makeCrowdedTrainingAndTestingSampleSets(trainSize, imSize, D, T, L);
    else
        [~,        vernierTestSet, ~,            vernierTestAnswers] = makeTrainingAndTestingSampleSets(       trainSize, imSize, D, T, L);
        [trainSet, crowdedTestSet, trainAnswers, crowdedTestAnswers] = makeCrowdedTrainingAndTestingSampleSets(trainSize, imSize, D, T, L);
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
    
    for run = 1:nRuns
        %% Build Training sets filtered through DNN, train and test for all conditions
        accuracies = zeros(length(readoutLayers),nExperiments+1);
        MSEs = zeros(length(readoutLayers),nExperiments+1);
        disp('starting hyperloop')
        N = 0;
        for currentLayer = readoutLayers
            
            N = N+1;
            
            %% Create a dnn-filtered train set
            disp('running train set through dnn');
            [netTrainSet, netTrainAnswers] = makeNetTrainSet(trainSet, trainAnswers, currentLayer, net);
            
            %% Train softmax classifier on the filtered train set
            disp('training softmax classifier')
            classifier = network(1,1,1,1,0,1);
            classifier.layers{1,1}.transferFcn = 'logsig';
            classifier.trainFcn = 'trainscg';
            classifier.divideFcn = 'dividerand';
            classifier.divideParam.trainRatio = 0.7;
            classifier.divideParam.valRatio = 0.15;
            classifier.divideParam.testRatio = 0.15;
            classifier.trainParam.epochs = 1000;
            classifier.trainParam.goal = 0;
            classifier.trainParam.time = inf;
            classifier.trainParam.min_grad = 1e-6;
            classifier.trainParam.max_fail = 10; %deafault 6
            classifier.trainParam.sigma = 5e-5; %default 5e-5
            classifier.trainParam.lambda = 5e-7;
            classifier.trainParam.showWindow = 1;
            [classifier, TR] = train(classifier,netTrainSet',netTrainAnswers','reduction',500);
            
            allClassifiers{currentStim,run,currentLayer} = classifier;
            allTrainResults{currentStim,run,currentLayer} = TR;
            
            %% Train set
            predictions     = classifier(netTrainSet(1:testSize,:)');
            answers         = netTrainAnswers(1:testSize)';
            accuracies(N,1) = accuracy(answers,predictions);
            MSEs(N,1)       = immse(answers,predictions);
            netTrainSet = [];
            netTrainAnswers = [];
            
            %% Test the classifier
            for k = 1:nExperiments
                if k == 1
                    testSet       = vernierTestSet;
                    testAnswers   = vernierTestAnswers;
                elseif k == 2
                    testSet       = crowdedTestSet;
                    testAnswers   = crowdedTestAnswers;
                else
                    testSet       = uncrowdedTestSets{   1,k-2};
                    testAnswers   = uncrowdedTestAnswers{1,k-2};
                end
                predictions       = getClassifierPredictions(testSet,net,currentLayer,classifier);
                accuracies(N,k+1) = accuracy(testAnswers-1,predictions);
                MSEs(N,k+1)       = immse(   testAnswers-1,predictions);
            end
            classifier = [];
            disp(['... for currentLayer = ', num2str(currentLayer)])
        end
        allAccuracies(currentStim, run, :, :) = accuracies;
        allMSEs(currentStim, run, :, :) = MSEs;
    end
end
%% Plot and save data


cd data
save('allAccuracies','allAccuracies')
save('allMSEs','allMSEs')
save('allClassifiers','allClassifiers')
save('allTrainResults','allTrainResults')
cd ..

% mean_accuracies = mean(allAccuracies, 2);
% std_accuracies = std(allAccuracies, 0, 2);
% 
% for stimSize = 1:length(stimSizes)
%     
%     figure(stimSize)
%     hold on
%     plotAccuracies(squeeze(mean_accuracies(stimSize, 1, :, :)), squeeze(std_accuracies(stimSize, 1, :, :)))
%     
% end


disp('I''m glad that''s done.')


