function [  ] = plotAccuracies( accuracies, std_accuracies )
%PLOTACCURACIES Plots accuracies
%   easy does it

for i = 1:size(accuracies,2)
    plot(accuracies(:,i))
%     errorbar(1:size(accuracies,1),accuracies(:,i),std_accuracies(:,i)/2,'.')
    hold on
end
legend('training set', 'vernier', 'crowded', 'uncrowded 1', 'uncrowded 2', 'uncrowded 3')
xlabel('layer')
ylabel('accuracy')
for i = 1:size(accuracies,2)
    errorbar(1:size(accuracies,1),accuracies(:,i),std_accuracies(:,i)/2,'.')
end

hold off

