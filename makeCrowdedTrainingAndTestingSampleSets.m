function [trainSet, testSet, trainAnswers, testAnswers] = makeCrowdedTrainingAndTestingSampleSets(nSamples, imSize, D, T, L)

%% Description
% Generate training and testing sample sets of left/right vernier stimuli
% Modifiy the parameters in the section below to your will
% Sample sets go in "/samples" (created in the current directory)

%% Parameters
%imSize = [50,50];     % number of [rows, columns] for each samples, in pixels % Henrik: before 28
%nSamples = 1000;      % number of samples for each vernier side in each set
dataType = 'uint8';   % '(u)int8,16,32,64' ; 'logical' ; 'double/single' ; etc.
%D = 1:10;             % various vernier offsets, in pixels
%T = 1:5;              % various vernier thickness, in pixels
%L = 5:12;             % various lengths for one vernier bar, in pixels

%% Directory management
[progDir,~,~] = fileparts(which(mfilename)); % get script directory
cd(progDir);                                 % go there just in case we're not already
sampleDir = [progDir filesep 'samples'];
if exist(sampleDir,'dir') == 0               % if there is no sample directory, make one
    mkdir('samples');
end

%% Load everything and randomize the training and testing set
[RTrainSet, RTestSet, LTrainSet, LTestSet] = createCrowdedSampleSets(imSize, nSamples, D, T, L, dataType);
trainSet = zeros(imSize(1), imSize(2), 2*nSamples);
trainSet(:,:,1:nSamples) = RTrainSet;
trainSet(:,:,nSamples+1:end) = LTrainSet;
trainAnswers = [ones(1,nSamples), 2*ones(1,nSamples)];
shuffleIndexes = randperm(length(trainSet));
trainSet(:,:,:) = trainSet(:,:,shuffleIndexes);
trainAnswers = trainAnswers(shuffleIndexes);

testSet = zeros(imSize(1), imSize(2), 2*nSamples);
testSet(:,:,1:nSamples) = RTestSet;
testSet(:,:,nSamples+1:end) = LTestSet;
testAnswers = [ones(1,nSamples), 2*ones(1,nSamples)];
shuffleIndexes = randperm(length(testSet));
testSet(:,:,:) = testSet(:,:,shuffleIndexes);
testAnswers = testAnswers(shuffleIndexes);

% %% Save everything in a separate directory
% cd(sampleDir)
% save('crowdedTrainSet', 'trainSet')
% save('crowdedTestSet', 'testSet')
% save('crowdedTrainAnswers', 'trainAnswers')
% save('crowdedTestAnswers', 'testAnswers')
% cd(progDir)

