function imdb = makeImdb(stims,answers,varargin)

opts = struct('trainRatio',0.7,...
'valRatio',0.15,...
'testRatio',0.15);
opts = vl_argparse(opts, varargin);

height = size(stims,1);
width = size(stims,2);
nSamples = size(stims,3);

data_ = zeros(227,227,3,nSamples);
for i = 1:nSamples
    im = repmat(stims(:,:,i)*225,[1,1,3]);
    im_ = single(im);
    im_ = imresize(im_, [227, 227]);
    im__ = single(im_);
    data_(:,:,:,i) = im__;
    clear im im_ im__
end
data = single(data_);

labels = (answers-1);

sets = 3*ones(1,nSamples);
trainInds = 1:round(opts.trainRatio*nSamples);
sets(trainInds) = ones(1,length(trainInds));
valInds = trainInds(end) + 1:...
    round((opts.trainRatio+opts.valRatio)*nSamples);
sets(valInds) = 2*ones(1,length(valInds));

images = struct('data',data,...
    'labels',labels,...
    'set',sets);
imdb = struct('images',images);


