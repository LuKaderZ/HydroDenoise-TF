%% 消融实验评估脚本（仅 test1）
clear; clc;

projectRoot = 'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF';
cleanDir = fullfile(projectRoot, 'data', 'ShipsEar', 'test1', 'clean');
noisyDir = fullfile(projectRoot, 'data', 'ShipsEar', 'test1', 'noisy');

% 三个消融变体的降噪音频目录
denoisedDirs = {
    fullfile(projectRoot, 'experiments', 'ablation', 'ablation1', 'denoised'), ...
    fullfile(projectRoot, 'experiments', 'ablation', 'ablation2', 'denoised'), ...
    fullfile(projectRoot, 'experiments', 'ablation', 'ablation3', 'denoised')
    };
modelNames = {'移除全局支路', '移除局部支路', '移除卷积增强'};

targetSNRs = [-15, -10, -5];
tolerance = 2.5;

files = dir(fullfile(cleanDir, '*.wav'));
[~, idx] = sort({files.name}); files = files(idx);

fprintf('\n========== 消融实验评估 ==========\n');

for m = 1:length(modelNames)
    if ~exist(denoisedDirs{m}, 'dir')
        fprintf('[跳过] %s\n', denoisedDirs{m});
        continue;
    end
    
    sisnri_cell = {[] [] []}; sdri_cell = {[] [] []};
    
    for f = 1:length(files)
        estFile = fullfile(denoisedDirs{m}, files(f).name);  % 文件名相同，直接匹配
        if ~exist(estFile, 'file'), continue; end
        
        clean = mean(audioread(fullfile(cleanDir, files(f).name)), 2);
        noisy = mean(audioread(fullfile(noisyDir, files(f).name)), 2);
        denoised = mean(audioread(estFile), 2);
        
        minLen = min([length(clean), length(noisy), length(denoised)]);
        clean = clean(1:minLen); noisy = noisy(1:minLen); denoised = denoised(1:minLen);
        
        actualSNR = 10 * log10(mean(clean.^2) / (mean((noisy - clean).^2) + 1e-10));
        [~, snrIdx] = min(abs(targetSNRs - actualSNR));
        if abs(targetSNRs(snrIdx) - actualSNR) > tolerance, continue; end
        
        sisnri_cell{snrIdx}(end+1) = compute_sisnr(denoised, clean) - compute_sisnr(noisy, clean);
        sdri_cell{snrIdx}(end+1)   = compute_sdr(denoised, clean)   - compute_sdr(noisy, clean);
    end
    
    fprintf('\n--- %s ---\n', modelNames{m});
    for i = 1:3
        fprintf('  %d dB: SI-SNRi=%.2f, SDRi=%.2f (n=%d)\n', ...
            targetSNRs(i), mean(sisnri_cell{i}), mean(sdri_cell{i}), length(sisnri_cell{i}));
    end
    fprintf('表4-2: SI-SNR=%.2f, SDR=%.2f\n', mean(cell2mat(sisnri_cell)), mean(cell2mat(sdri_cell)));
end

function s = compute_sisnr(est, ref)
est=est(:)-mean(est(:)); ref=ref(:)-mean(ref(:));
s_target = (est'*ref)/(ref'*ref+1e-8)*ref; e=est-s_target;
s=10*log10((s_target'*s_target)/(e'*e+1e-8)+1e-8);
end
function s = compute_sdr(est, ref)
e=est(:)-ref(:); s=10*log10((ref(:)'*ref(:))/(e'*e+1e-8)+1e-8);
end