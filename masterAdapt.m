%% Parameters
loadNet = 1;
createSets = 1;
feedNet = 1;
minibatchSize = 1000;
Ns = 1:21; % layer of enquiry, out of 21
nSamples = 5000;
imSize = [227, 227];
D = 1:5; % 1:10
T = 1:3; % 1:5
L = 3:7; % 5:12
nUncrowded = 1:3;
nExperiments = 1 + 1 + length(nUncrowded);
testSize = 500; % round(0.3*nSamples);
seed = 1995;
%% Dependant parameters
nMinibatches = nSamples/minibatchSize;
rng(seed);
%% Load net
if loadNet
disp('loading net');
net = load('../nets/imagenet-caffe-alex.mat') ; % Load the network
net = vl_simplenn_tidy(net) ; % update and fill in values
close all
end
%% Creating sets
if createSets
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
%%

tic

accuracies = zeros(length(Ns),nExperiments+1);
MSEs = zeros(length(Ns),nExperiments+1);

compareAccuracies = zeros(length(Ns),nMinibatches);
compareMSEs = zeros(length(Ns),nMinibatches);

disp('starting hyperloop')
for N = Ns
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
        
        if k == 1
            B = 1;
            for b = 1:minibatchSize:nSamples-minibatchSize
            trainSet = normalTrainSet(:,:,b:b+minibatchSize-1);
            trainAnswers = normalTrainAnswers(b:b+minibatchSize-1);
            if feedNet
            tic
            disp('running train set through dnn');
            [netTrainSet, netTrainAnswers] = makeNetTrainSet(trainSet, trainAnswers, N, net);
            toc
            end
            
            % Training softmax classifier
            tic
            disp('training softmax classifier')
            x = netTrainSet';
            t = netTrainAnswers';
            
classifier = network(1,1,1,1,0,1);
classifier.layers{1,1}.transferFcn = 'logsig';
classifier.adaptFcn = 'trains';
%classifier.trainFcn = 'trains';
classifier.inputWeights{1,1}.learnFcn = 'learngd';
classifier.biases{1,1}.learnFcn = 'learngd';
%classifier.trainParam.showWindow = 1;

            classifier.divideFcn = 'dividerand';
            classifier.divideParam.trainRatio = 0.7;
            classifier.divideParam.valRatio = 0.15;
            classifier.divideParam.testRatio = 0.15;
            
            [classifier, Y, E, ~, ~, tr] = adapt(classifier,x,t);
            %[classifier, tr] = trains(classifier,x,t);
            y = classifier(netTrainSet');
            
            predictions = y; %(1:testSize);
            answers = t; %(1:testSize);
            accuracies(N,1) = accuracy(answers,predictions);
            MSEs(N,1) = immse(answers,predictions);

            compareAccuracies(N,B) = accuracy(y,t);
            compareMSEs(N,B) = immse(y,t);
            B = B +1;
            
            cd data
            save('accuracies','accuracies')
            save('MSEs','MSEs')
            save('compareAccuracies','compareAccuracies')
            save('compareMSEs','compareMSEs')
            cd ..
            
%             figure
%             plotperform(tr)
            
            toc
            end
        end
        
        % Testing the classifier
        tic
        predictions = getClassifierPredictions(testSet,net,N,classifier);
        if k == 1
            accuracies(N,k+1) = accuracy(testAnswers-1,predictions);
            MSEs(N,k+1) = immse(testAnswers-1,predictions);
            cd data
            save('accuracies','accuracies')
            save('MSEs','MSEs')
            cd ..
            toc
        elseif k == 2
            accuracies(N,k+1) = accuracy(testAnswers-1,predictions);
            MSEs(N,k+1) = immse(testAnswers-1,predictions);
            cd data
            save('accuracies','accuracies')
            save('MSEs','MSEs')
            cd ..
            toc
        else
            accuracies(N,k+1) = accuracy(testAnswers-1,predictions);
            MSEs(N,k+1) = immse(testAnswers-1,predictions);
            cd data
            save('accuracies','accuracies')
            save('MSEs','MSEs')
            cd ..
            toc
        end
    end
    clear classifier x t y netTrainSet netTrainAnswers
    disp(['... for N = ', num2str(N)])
end
disp('I''m glad that''s done.')