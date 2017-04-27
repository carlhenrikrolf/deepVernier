function accuracies = svmlayer0(nSamples)
%% Parameters
%nSamples = 300;
imSize = [50, 160]; %min = [45, 155]
D = 1:10; % 1:10
T = 1:5; % 1:5
L = 5:12; % 5:12
nUncrowded = 1;
nExperiments = 1 + 1 + length(nUncrowded);
testSize = round(0.3*nSamples);
% seed = 1995;
% rng(seed);
%%
if 1
    tic
    disp('creating stimuli');
    [normalTrainSet, normalTestSet, normalTrainAnswers, normalTestAnswers] = ...
        makeTrainingAndTestingSampleSets(nSamples, imSize, D, T, L);
    [crowdedTrainSet, crowdedTestSet, crowdedTrainAnswers, crowdedTestAnswers] = ...
        makeCrowdedTrainingAndTestingSampleSets(nSamples, imSize, D, T, L);
    
    uncrowdedTestSets = cell(1,length(nUncrowded));
    uncrowdedTestAnswers = cell(1,length(nUncrowded));
    uncrowdedTrainSets = cell(1,length(nUncrowded));
    uncrowdedTrainAnswers = cell(1,length(nUncrowded));
    for i = 1:length(nUncrowded)
        [uncrowdedTrainSets{1,i}, uncrowdedTestSets{1,i}, uncrowdedTrainAnswers{1,i}, uncrowdedTestAnswers{1,i}] = ...
            makeUncrowdedTrainingAndTestingSampleSets(nSamples, imSize, D, T, L, nUncrowded(i));
    end
end
%%
trainSet = crowdedTrainSet;
trainAnswers = crowdedTrainAnswers;

accuracies = zeros(1,1);
MSEs = zeros(1,1);
%%
disp('preprocessing stimuli')
temp = trainSet(:,:,i);
im = repmat(temp(:,:)*255,[1,1,3]);
im_ = single(im) ;
im_ = imresize(im_, [227 227]) ;
xlen = length(im_(:));
x = zeros(xlen,2*nSamples);
x(:,1) = im_(:);
clear temp im im_
for i = 2:2*nSamples
    temp = trainSet(:,:,i);
    im = repmat(temp(:,:)*255,[1,1,3]);
    im_ = single(im) ;
    im_ = imresize(im_, [227 227]) ;
    x(:,i) = im_(:);
    clear temp im im_
end
t = trainAnswers-1;
%%
disp('training classifier')
options = statset('MaxIter',1e9); %default 15000
classifier = svmtrain(x',t,'options',options);


%%
disp('doing experiments')
for k = 0:nExperiments
    if k == 0
        testSet = trainSet(:,:,1:testSize);
        testAnswers = trainAnswers(1:testSize);
    elseif k == 1
        testSet = normalTestSet(:,:,1:testSize);
        testAnswers = normalTestAnswers(1:testSize);
    elseif k == 2
        testSet = crowdedTestSet(:,:,1:testSize);
        testAnswers = crowdedTestAnswers(1:testSize);
    else
        testSet = uncrowdedTestSets{1,k-2}(:,:,1:testSize);
        testAnswers = uncrowdedTestAnswers{1,k-2}(1:testSize);
    end
    x = zeros(xlen,testSize);
    for i = 1:testSize
        temp = testSet(:,:,i);
        im = repmat(temp(:,:)*255,[1,1,3]);
        im_ = single(im) ;
        im_ = imresize(im_, [227 227]) ;
        x(:,i) = im_(:);
        clear temp im im_
    end
    t = testAnswers-1;
    predictions = svmclassify(classifier,x')';
    accuracies(1,k+1) = accuracy(t,predictions);
    MSEs(1,k+1) = immse(t,predictions);
end
%% Classifiers
    function classifier = NNClassifier(x,t,hiddenLayerSize)
        classifier = hiddenClassifier(hiddenLayerSize);
        [classifier, TR] = train(classifier,x,t,'reduction',500);
        %plotperform(TR)
        function classifier = hiddenClassifier(hiddenLayerSize)
            %hiddenLayerSize = 100;
            classifier = patternnet(hiddenLayerSize);
            %classifier.trainFcn = 'trainscg';
            classifier.divideFcn = 'dividerand';
            classifier.divideParam.trainRatio = 0.7;
            classifier.divideParam.valRatio = 0.15;
            classifier.divideParam.testRatio = 0.15;
%             classifier.trainParam.epochs = 1500;
%             classifier.trainParam.goal = 0;
%             classifier.trainParam.time = inf;
%             classifier.trainParam.min_grad = 1e-6;
%             classifier.trainParam.max_fail = 6; %default 6
%             classifier.trainParam.sigma = 5e-5; %default 5e-5
%             classifier.trainParam.lambda = 5e-7;
            classifier.trainParam.showWindow = 1;
        end
        
        function classifier = softmaxClassifier
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
            classifier.trainParam.max_fail = 20;
            classifier.trainParam.sigma = 5e-5; %default 5e-5
            classifier.trainParam.lambda = 5e-7;
            classifier.trainParam.showWindow = 1;
        end
    end
end