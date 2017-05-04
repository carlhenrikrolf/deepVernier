function out = general_mine

%setup;

imdb = getImdb;

% % Get net
% f=1/100 ;
% net.layers = {} ;
% net.layers{end+1} = struct('type', 'conv', ...
%     'weights', {{f*randn(2,2,1,4, 'single'), []}}, ...
%     'stride', 1, ...
%     'pad', 0) ;
% net.layers{end+1} = struct('type', 'pool', ...
%     'method', 'max', ...
%     'pool', [2 2], ...
%     'stride', 2, ...
%     'pad', 0) ;
% net.layers{end+1} = struct('type', 'softmaxloss') ;

net = load('../nets/imagenet-caffe-alex.mat'); % Load the network

net.meta.trainOpts.learningRate = 0.001 ;
net.meta.trainOpts.numEpochs = 20 ;
net.meta.trainOpts.batchSize = 1 ;

% Fill in defaul values
net = vl_simplenn_tidy(net) ;

% run(fullfile(fileparts(mfilename('fullpath')),...
%     '..', '..', 'matlab', 'vl_setupnn.m')) ;

% Train
trainfn = @cnn_train;
opts.train = struct('gpus',[],...
    'backPropDepth', 1) ;
[net, info] = trainfn(net, imdb, getBatch, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

res = vl_simplenn(net,imdb.images.data(:,:,:,1));

out = struct('net', net,...
    'info',info,...
    'imdb',imdb,...
    'res',res);

    function imdb = getImdb
        data = zeros(3,3,1,3,'single');
        labels = zeros(1,3);
        set_(1,3) = 1;
        data(:,:,1,1) = [1 0 0
                         0 1 0
                         0 0 0];
        labels(1) = 0;
        set_(1) = 1;
        data(:,:,1,2) = [0 1 0
                         1 0 0
                         0 0 0];
        labels(2) = 1;
        set_(2) = 1;
        data(:,:,1,3) = [0 1 0
                         1 0 0
                         0 0 0];
        labels(3) = 1;
        set_(3) = 3;
        
        data_mean = mean(data,4);
        
        clear data
        data = zeros(227,227,3,3,'single');
        
        images = struct('data',data,...
            'labels',labels,...
            'set',set_);
        imdb = struct('images',images);
    end

%     function fn = getBatch
%         fn = @(x,y) getSimpleNNBatch(x,y) ;
%     end
% 
%     function [images, labels] = getSimpleNNBatch(imdb, batch)
%         images = imdb.images.data(:,:,:,batch) ;
%         labels = imdb.images.labels(1,batch) ;
%     end
end