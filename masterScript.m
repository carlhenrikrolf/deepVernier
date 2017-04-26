%% Load net
if 1
disp('loading net');
net = load('../nets/imagenet-caffe-alex.mat') ; % Load the network
net = vl_simplenn_tidy(net) ; % update and fill in values
close all
end

%% Parameters
Ns = 1:2; % layer of enquiry, out of 21
nSamples = 500;
imSize = [227, 227];
D = 1:5; % 1:10
T = 1:3; % 1:5
L = 3:7; % 5:12
nUncrowded = 1:3;
nExperiments = 1 + 1 + length(nUncrowded);
testSize = round(0.3*nSamples);
seed = 1995;
rng(seed);

%% Creating sets
if 1
tic
disp('creating stimuli');
[normalTrainSet, normalTestSet, normalTrainAnswers, normalTestAnswers] = ...
    makeTrainingAndTestingSampleSets(nSamples, imSize, D, T, L);
[~, crowdedTestSet, ~, crowdedTestAnswers] = ...
    makeCrowdedTrainingAndTestingSampleSets(nSamples, imSize, D, T, L);

uncrowdedTestSets = cell(1,length(nUncrowded));
uncrowdedTestAnswers = cell(1,length(nUncrowded));
for i = 1:length(nUncrowded)
    [~, uncrowdedTestSets{1,i}, ~, uncrowdedTestAnswers{1,i}] = ...
    makeUncrowdedTrainingAndTestingSampleSets(nSamples, imSize, D, T, L, nUncrowded(i));
end
end

% len = length(normalTrainAnswers) + length(crowdedTrainAnswers);
% shuffling = randperm(len);
% tempSet = cat(3,normalTrainSet,crowdedTrainSet);
% tempSet = tempSet(shuffling);
% tempAnswers = cat(2,normalTrainAnswers,crowdedTrainAnswers);
% tempAnswers = tempAnswers(shuffling);

trainSet = normalTrainSet;
trainAnswers = normalTrainAnswers;

%%
tic

accuracies = zeros(length(Ns),nExperiments+1);
MSEs = zeros(length(Ns),nExperiments+1);

compareAccuracies = zeros(length(Ns),1);
compareMSEs = zeros(length(Ns),1);

disp('starting hyperloop')
parfor N = Ns
    for k = 1:nExperiments
        if k == 1
            testSet = normalTestSet(:,:,1:testSize);
            testAnswers = normalTestAnswers(1:testSize);
        elseif k == 2
            testSet = crowdedTestSet(:,:,1:testSize);
            testAnswers = crowdedTestAnswers(1:testSize);
        else
            testSet = uncrowdedTestSets{1,k-2}(:,:,1:testSize);
            testAnswers = uncrowdedTestAnswers{1,k-2}(1:testSize);
        end
        
        % %%
        % processedTestSet = processImages(testSet, net);
        % processedTrainSet = processImages(trainSet, net);
        % toc
        % %% Run through pretrained net
        % tic
        % disp('running through pretrained net')
        % for i = 1:length(testSet)
        %     res
        %
        % toc
        %%
        if k == 1
%             tic
%             disp('running train set through dnn');
            [netTrainSet, netTrainAnswers] = makeNetTrainSet(trainSet, trainAnswers, N, net);
%             toc
        end
        %% Training softmax classifier
        if k == 1
%             tic
%             disp('training softmax classifier')
            x = netTrainSet';
            t = netTrainAnswers';
            
%             classifier = trainSoftmaxLayer(x,t,...
%                 'LossFunction', 'mse',...
%                 'MaxEpochs', 1,...
%                 'ShowProgressWindow',1,...
%                 'TrainingAlgorithm','trainscg');
%             disp('first epoch done');
classifier = network(1,1,1,1,0,1);
classifier.layers{1,1}.transferFcn = 'logsig';
classifier.trainFcn = 'trainscg';
            classifier.divideFcn = 'dividerand';
            classifier.divideParam.trainRatio = 0.7;
            classifier.divideParam.valRatio = 0.15;
            classifier.divideParam.testRatio = 0.15;
            classifier.trainParam.epochs = 100;
            classifier.trainParam.goal = 0;
            classifier.trainParam.time = inf;
            classifier.trainParam.min_grad = 1e-6;
            classifier.trainParam.max_fail = 6;
            classifier.trainParam.sigma = 5e-4; %default 5e-5
            classifier.trainParam.lambda = 5e-7;
            classifier.trainParam.viewWindow = 0;
            [classifier, TR] = train(classifier,x,t);
            
            
            y = classifier(netTrainSet');
            
            predictions = y(1:testSize);
            answers = t(1:testSize);
%             accuracies(N,1) = accuracy(answers,predictions);
%             MSEs(N,1) = immse(answers,predictions);

            compareAccuracies(N) = accuracy(y,t);
            compareMSEs(N) = immse(y,t);
            
%             cd data
%             save('accuracies','accuracies')
%             save('MSEs','MSEs')
%             save('compareAccuracies','compareAccuracies')
%             save('compareMSEs','compareMSEs')
%             cd ..
            
%             figure
%             plotperform(TR)
            
%             figure
%             plotconfusion(t,y)
%             toc
        end
        
        %% Testing the classifier
%         tic
        predictions = getClassifierPredictions(testSet,net,N,classifier);
        accuracies(N,k) = accuracy(testAnswers-1,predictions);
            MSEs(N,k) = immse(testAnswers-1,predictions);
%         if k == 1
%             accuracies(N,k+1) = accuracy(testAnswers-1,predictions);
%             MSEs(N,k+1) = immse(testAnswers-1,predictions);
% %             cd data
% %             save('accuracies','accuracies')
% %             save('MSEs','MSEs')
% %             cd ..
% %             toc
%         elseif k == 2
%             accuracies(N,k+1) = accuracy(testAnswers-1,predictions);
%             MSEs(N,k+1) = immse(testAnswers-1,predictions);
% %             cd data
% %             save('accuracies','accuracies')
% %             save('MSEs','MSEs')
% %             cd ..
% %             toc
%         else
%             accuracies(N,k+1) = accuracy(testAnswers-1,predictions);
%             MSEs(N,k+1) = immse(testAnswers-1,predictions);
% %             cd data
% %             save('accuracies','accuracies')
% %             save('MSEs','MSEs')
% %             cd ..
% %             toc
%         end
    end
%     clear classifier x t y netTrainSet netTrainAnswers
    disp(['... for N = ', num2str(N)])
end
disp('I''m glad that''s done.')
