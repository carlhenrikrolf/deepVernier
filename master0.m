nSamples = 6000:2000:16000;
N = length(nSamples);
results = cell(1,N);
parfor n = 1:N
    results{1,n} = layer0(nSamples(n));
end
results