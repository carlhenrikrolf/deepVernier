
%% Make training and testing sample sets for verniers of different offsets/thickness/lengths/positions
function [RTrainSet, RTestSet, LTrainSet, LTestSet] = createSampleSets(imSize,nSamples,D,T,L,dataType)

    % Create all possible vernier filters (right ones, fliplr on it for left ones)
    filters = cell(length(D),length(T),length(L));
    for d = D
        for t = T
            for l = L

                filters{d-min(D)+1,t-min(T)+1,l-min(L)+1} = createVernierFilter(d,t,l,dataType);

            end
        end
    end
    filters = filters(:); % make one vector of vernier filters

    % Fill the cells with samples containing randomized verniers
    RTrainSet = zeros(imSize(1),imSize(2),nSamples);
    RTestSet = zeros(imSize(1),imSize(2),nSamples);
    LTrainSet = zeros(imSize(1),imSize(2),nSamples);
    LTestSet = zeros(imSize(1),imSize(2),nSamples);
    for i = 1:nSamples

        RTrainSet(:,:,i) = createSample(imSize, filters{randi(length(filters))}, dataType, 'train');
        RTestSet(:,:,i) = createSample(imSize, filters{randi(length(filters))}, dataType, 'test');
        LTrainSet(:,:,i) = createSample(imSize, fliplr(filters{randi(length(filters))}), dataType, 'train');
        LTestSet(:,:,i) = createSample(imSize, fliplr(filters{randi(length(filters))}), dataType, 'test');
        
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

%% Make a sample to fill a training or a testing set
function sample = createSample(imSize, filter, dataType, sampleType)

    % Create a basis image of the right size/type
    if strcmp(dataType,'logical')
        sample = false(imSize, dataType);
    else
        sample = zeros(imSize, dataType);
    end

    % Choose the position where to add a vernier patch (top-left corner)
    if strcmp(sampleType,'train')
        row = randi(imSize(1)-size(filter,1));
        col = randi(imSize(2)-size(filter,2));
    else
        row = ceil((imSize(1) - size(filter,1))/2) + 1;
        col = ceil((imSize(2) - size(filter,2))/2);
    end

    % Draw the vernier patch
    sample(row:row+size(filter,1)-1, col:col+size(filter,2)-1) = filter;

end

