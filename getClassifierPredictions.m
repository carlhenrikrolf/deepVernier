function predictions = getClassifierPredictions(testSet,net,N,classifier)

len = size(testSet,3);
predictions = zeros(1,len);
for i = 1:len
    im = repmat(testSet(:,:,i)*255,[1,1,3]);
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    %im_ = im_ - net.meta.normalization.averageImage ;
    res = vl_simplenn(net, im_);
    predictions(i) = classifier(res(N).x(:));
end