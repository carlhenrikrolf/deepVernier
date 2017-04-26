%% Make one crowded sample set for verniers and flankers with various features
function [RCrowdedTrainSet, RCrowdedTestSet, LCrowdedTrainSet, LCrowdedTestSet] = createCrowdedSampleSets(imSize,nSamples,D,T,L,dataType)

% Create all possible vernier filters (right ones, fliplr on it for left ones)
vernierFilters = cell(length(D),length(T),length(L));
squareFilters = cell(length(D),length(T),length(L));
for d = D
    for t = T
        for l = L

            vernierFilters{d-min(D)+1,t-min(T)+1,l-min(L)+1} = createVernierFilter(d,t,l,dataType);
            squareFilters{d-min(D)+1,t-min(T)+1,l-min(L)+1} = createSquareFilter(d,t,l,dataType);

        end
    end
end
vernierFilters = vernierFilters(:); % make one vector of vernier filters
squareFilters = squareFilters(:); % make one vector of square filters

% Fill the cells with samples containing randomized verniers
LCrowdedTestSet = zeros(imSize(1),imSize(2),nSamples);
RCrowdedTestSet = zeros(imSize(1),imSize(2),nSamples);

LCrowdedTrainSet = zeros(imSize(1),imSize(2),nSamples);
RCrowdedTrainSet = zeros(imSize(1),imSize(2),nSamples);

for i = 1:nSamples
    
    randomIndex = randi(length(vernierFilters));
    vernierFilter = vernierFilters{randomIndex};
    squareFilter = squareFilters{randomIndex};
    RCrowdedTrainSet(:,:,i) = createCrowdedSample(imSize, vernierFilter, squareFilter, dataType, 'train');
    
    randomIndex = randi(length(vernierFilters));
    vernierFilter = vernierFilters{randomIndex};
    squareFilter = squareFilters{randomIndex};
    RCrowdedTestSet(:,:,i) = createCrowdedSample(imSize, vernierFilter, squareFilter, dataType, 'test');
    
    randomIndex = randi(length(vernierFilters));
    vernierFilter = vernierFilters{randomIndex};
    squareFilter = squareFilters{randomIndex};
    LCrowdedTrainSet(:,:,i) = createCrowdedSample(imSize, fliplr(vernierFilter), squareFilter, dataType, 'train');
    
    randomIndex = randi(length(vernierFilters));
    vernierFilter = vernierFilters{randomIndex};
    squareFilter = squareFilters{randomIndex};
    LCrowdedTestSet(:,:,i) = createCrowdedSample(imSize, fliplr(vernierFilter), squareFilter, dataType, 'test');
end

end

%% Draw a parametrized (offset/thickness/length/type) vernier filter
function vernierFilter = createVernierFilter(d, t, l, dataType)

% Create the basis filter (filled with zeros)
if strcmp(dataType,'logical')
    vernierFilter = false(2*l, 2*t+d, dataType);
else
    vernierFilter = zeros(2*l, 2*t+d, dataType);
end

% Draw the lines (ones)
vernierFilter(1:l, 1:t) = 1;
vernierFilter(l+1:end, t+d+1:end) = 1;

end

function squareFilter = createSquareFilter(d, t, l, dataType)

% Find the size of the square
gapSize = round((sqrt(2)-1)/2*max(2*t+d,2*l));
a = max(2*t+d, 2*l) + 2*gapSize;

% Create the basis filter (filled with zeros)
if strcmp(dataType,'logical')
    squareFilter = false(a+2*t, a+2*t, dataType); % Henrik: Added '2*' in 'a+t'
else
    squareFilter = zeros(a+2*t, a+2*t, dataType); % Henrik: Added '2*' in 'a+t'
end

% Draw the square
squareFilter(1:t,:) = 1;
squareFilter(end-t+1:end,:) = 1;
squareFilter(:,1:t) = 1;
squareFilter(:,end-t+1:end) = 1;

end

%% Make a sample to fill a training or a testing set
function sample = createCrowdedSample(imSize, vernierFilter, squareFilter, dataType, sampleType)

% Create a basis image of the right size/type
if strcmp(dataType,'logical')
    sample = false(imSize, dataType);
else
    sample = zeros(imSize, dataType);
end

% Choose the position square (top-left corner)
a = size(squareFilter,1);
rowS = randi(imSize(1)-a);
colS = randi(imSize(2)-a);

% Set the position of the vernier (top-left corner)
nRowV = size(vernierFilter,1);
nColV = size(vernierFilter,2);
rowV = rowS + round((a-nRowV)/2);
colV = colS + round((a-nColV)/2);

% Draw the vernier and the square patches
sample(rowS:rowS+size(squareFilter,1)-1, colS:colS+size(squareFilter,2)-1) = squareFilter;
sample(rowV:rowV+size(vernierFilter,1)-1, colV:colV+size(vernierFilter,2)-1) = vernierFilter;

end

