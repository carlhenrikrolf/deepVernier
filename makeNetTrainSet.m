function [DNNTrainSet, DNNTrainAnswers] = makeNetTrainSet(trainSet, trainAnswers, N, net)

im = repmat(trainSet(:,:,1)*255,[1,1,3]);
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
%im_ = im_ - net.meta.normalization.averageImage ;

res = vl_simplenn(net, im_);
outSize = length(res(N).x(:));
DNNTrainSet = zeros(size(trainSet,3),outSize);
DNNTrainAnswers = (trainAnswers-1)';


for i = 1:size(trainSet,3)
    im = repmat(trainSet(:,:,i)*255,[1,1,3]);
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net.meta.normalization.averageImage ;
    res = vl_simplenn(net, im_);
    DNNTrainSet(i,:) = single(res(N).x(:));
end