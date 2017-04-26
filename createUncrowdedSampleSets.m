%% Make one crowded sample set for verniers and flankers with various features
function [RCrowdedTrainSet, RCrowdedTestSet, LCrowdedTrainSet, LCrowdedTestSet] = createUncrowdedSampleSets(imSize,nSamples,D,T,L,dataType,nFlankerPairs)

% Create all possible vernier filters (right ones, fliplr on it for left ones)
vernierFilters = cell(length(D),length(T),length(L));
squareFilters = cell(length(D),length(T),length(L));
flankersFilters = cell(length(D),length(T),length(L));
for d = D
    for t = T
        for l = L
            
            vernierFilters{d-min(D)+1,t-min(T)+1,l-min(L)+1} = createVernierFilter(d,t,l,dataType);
            squareFilters{d-min(D)+1,t-min(T)+1,l-min(L)+1} = createSquareFilter(d,t,l,dataType);
            flankersFilters{d-min(D)+1,t-min(T)+1,l-min(L)+1} = createFlankersFilter(d,t,l,dataType, nFlankerPairs);
        end
    end
end
vernierFilters = vernierFilters(:); % make one vector of vernier filters
squareFilters = squareFilters(:); % make one vector of square filters
flankersFilters = flankersFilters(:);

% Fill the cells with samples containing randomized verniers
LCrowdedTestSet = zeros(imSize(1),imSize(2),nSamples);
RCrowdedTestSet = zeros(imSize(1),imSize(2),nSamples);

LCrowdedTrainSet = zeros(imSize(1),imSize(2),nSamples);
RCrowdedTrainSet = zeros(imSize(1),imSize(2),nSamples);

for i = 1:nSamples
    
    % continue here with updating flankersFilters
    randomIndex = randi(length(vernierFilters));
    vernierFilter = vernierFilters{randomIndex};
    squareFilter = squareFilters{randomIndex};
    flankersFilter = flankersFilters{randomIndex};
    RCrowdedTrainSet(:,:,i) = createUncrowdedSample(imSize, vernierFilter, squareFilter, flankersFilter, dataType, nFlankerPairs, 'train');
    
    randomIndex = randi(length(vernierFilters));
    vernierFilter = vernierFilters{randomIndex};
    squareFilter = squareFilters{randomIndex};
    flankersFilter = flankersFilters{randomIndex};
    RCrowdedTestSet(:,:,i) = createUncrowdedSample(imSize, vernierFilter, squareFilter, flankersFilter, dataType, nFlankerPairs, 'test');
    
    randomIndex = randi(length(vernierFilters));
    vernierFilter = vernierFilters{randomIndex};
    squareFilter = squareFilters{randomIndex};
    flankersFilter = flankersFilters{randomIndex};
    LCrowdedTrainSet(:,:,i) = createUncrowdedSample(imSize, fliplr(vernierFilter), squareFilter, flankersFilter, dataType, nFlankerPairs, 'train');
    
    randomIndex = randi(length(vernierFilters));
    vernierFilter = vernierFilters{randomIndex};
    squareFilter = squareFilters{randomIndex};
    flankersFilter = flankersFilters{randomIndex};
    LCrowdedTestSet(:,:,i) = createUncrowdedSample(imSize, fliplr(vernierFilter), squareFilter, flankersFilter, dataType, nFlankerPairs, 'test');
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

% Find the inner size of the square
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

function flankersFilter = createFlankersFilter(d, t, l, dataType, nFlankerPairs)
% Find the inner size of the square
gapSize = round((sqrt(2)-1)/2*max(2*t+d,2*l));
a = max(2*t+d, 2*l) + 2*gapSize;

% Find the separation between flankers
s = round(0.25*(a + 2*t));

% Create the basis filter (filled with zeros)
h = a + 2*t;
w = h*(1 + 2*nFlankerPairs) + 2*nFlankerPairs*s;
if strcmp(dataType,'logical')
    flankersFilter = false(h, w, dataType);
else
    flankersFilter = zeros(h, w, dataType);
end

% Draw the flankers
m = 0;
for n = 1:nFlankerPairs
    % left flankers
    flankersFilter(1:t, m + (1:h)) = 1;
flankersFilter(end-t+1:end, m + (1:h)) = 1;
flankersFilter(:, m + (1:t)) = 1;
flankersFilter(:,m + (h-t+1:h)) = 1;

% right flankers 
    flankersFilter(1:t, end - (m + (1:h)) + 1) = 1;
flankersFilter(end-t+1:end, end - (m + (1:h)) + 1) = 1;
flankersFilter(:, end - (m + (1:t)) + 1) = 1;
flankersFilter(:, end - (m + (h-t+1:h)) + 1) = 1;

% update m
m = m + h + s;
    
end

end

%% Make a sample to fill a training or a testing set
function sample = createUncrowdedSample(imSize, vernierFilter, squareFilter, flankersFilter, dataType, nFlankerPairs, sampleType)

% Create a basis image of the right size/type
if strcmp(dataType,'logical')
    sample = false(imSize, dataType);
else
    sample = zeros(imSize, dataType);
end

% Choose the position of the flankers (top-left corner)
[h, w] = size(flankersFilter);
rowF = randi(imSize(1) - h);
colF = randi(imSize(2) - w);

% Set the position of the square (top-left corner)
s = round(0.25*h);
m = nFlankerPairs*(h+s);
rowS = rowF;
colS = colF+m;

% Set the position of the vernier (top-left corner)
nRowV = size(vernierFilter,1);
nColV = size(vernierFilter,2);
rowV = rowS + round((h-nRowV)/2);
colV = colS + round((h-nColV)/2);

% Draw the vernier and the square patches
sample(rowF:rowF+size(flankersFilter,1)-1, colF:colF+size(flankersFilter,2)-1) = flankersFilter;
sample(rowS:rowS+size(squareFilter,1)-1, colS:colS+size(squareFilter,2)-1) = squareFilter;
sample(rowV:rowV+size(vernierFilter,1)-1, colV:colV+size(vernierFilter,2)-1) = vernierFilter;

end

