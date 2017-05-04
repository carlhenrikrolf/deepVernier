function [acc,mse] = getResult(net,stims,answers)
testSize = size(stims,4);
x = zeros(1,testSize);
for i = 1:testSize
    res = vl_simplenn(net,stims(:,:,:,i));
    x(i) = res(end).x;
    clear res % ?
end
answers = answers - 1;
acc = accuracy(x,answers);
mse = immse(x,answers);