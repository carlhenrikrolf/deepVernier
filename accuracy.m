function out = accuracy(x,y)
% binary classification accuracy

% check for errors
if sum(x<0)>0
    warning('negative element, -s in 1st input (x)')
end
if sum(y<0)>0
    warning('negative element, -s in 2nd input (y)')
end
if sum(x>1)>0
    warning('> 1 element, -s in 1st input (x)')
end
if sum(y>1)>0
    warning('> 1 element, -s in 2nd input (y)')
end
if length(x) ~= length(y)
    warning(['input sizes not the same, 1st (x) = ', num2str(length(x)),...
        ' and 2nd (y) = ', num2str(length(y))])
end

% the function
nPredictions = length(x);
acc0s = sum((x < 0.5) & (y < 0.5));
acc1s = sum((x >= 0.5) & (y >= 0.5));
out = (acc0s + acc1s)/nPredictions;