%% Conv-TasNet 评估（test1 + DeepShip）
clear; clc;

projectRoot = 'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF';

% ==================== 处理 ShipsEar test1 ====================
cleanDir = fullfile(projectRoot, 'data', 'ShipsEar', 'test1', 'clean');
noisyDir = fullfile(projectRoot, 'data', 'ShipsEar', 'test1', 'noisy');
denoisedDir = fullfile(projectRoot, 'experiments', 'conv_tasnet', 'estimates', 'tt_test1');

targetSNRs = [-15, -10, -5];
tolerance = 2.5;

files = dir(fullfile(cleanDir, '*.wav'));
[~, idx] = sort({files.name}); files = files(idx);

sisnri_cell = {[] [] []}; sdri_cell = {[] [] []};

for f = 1:length(files)
    estFile = fullfile(denoisedDir, sprintf('%06d_sph_est.wav', f-1));
    if ~exist(estFile, 'file'), continue; end

    clean = mean(audioread(fullfile(cleanDir, files(f).name)), 2);
    noisy = mean(audioread(fullfile(noisyDir, files(f).name)), 2);
    denoised = mean(audioread(estFile), 2);

    minLen = min([length(clean), length(noisy), length(denoised)]);
    clean = clean(1:minLen); noisy = noisy(1:minLen); denoised = denoised(1:minLen);

    actualSNR = 10 * log10(mean(clean.^2) / (mean((noisy - clean).^2) + 1e-10));
    [~, idx] = min(abs(targetSNRs - actualSNR));
    if abs(targetSNRs(idx) - actualSNR) > tolerance, continue; end

    sisnri_cell{idx}(end+1) = compute_sisnr(denoised, clean) - compute_sisnr(noisy, clean);
    sdri_cell{idx}(end+1)   = compute_sdr(denoised, clean)   - compute_sdr(noisy, clean);
end

fprintf('Conv-TasNet test1:\n');
for i = 1:3
    fprintf('  %d dB: SI-SNRi=%.2f, SDRi=%.2f (n=%d)\n', ...
        targetSNRs(i), mean(sisnri_cell{i}), mean(sdri_cell{i}), length(sisnri_cell{i}));
end
fprintf('表4-1 ShipsEar: SI-SNRi=%.2f, SDRi=%.2f\n', mean(cell2mat(sisnri_cell)), mean(cell2mat(sdri_cell)));

% ==================== 处理 DeepShip ====================
cleanDir = fullfile(projectRoot, 'data', 'DeepShip', 'test', 'clean');
noisyDir = fullfile(projectRoot, 'data', 'DeepShip', 'test', 'noisy');
denoisedDir = fullfile(projectRoot, 'experiments', 'conv_tasnet', 'estimates', 'tt_DeepShip');

if ~exist(denoisedDir, 'dir')
    fprintf('\nDeepShip 增强音频目录不存在，跳过。\n');
else
    files = dir(fullfile(cleanDir, '*.wav'));
    [~, idx] = sort({files.name}); files = files(idx);

    sisnri_list = []; sdri_list = [];

    for f = 1:length(files)
        estFile = fullfile(denoisedDir, sprintf('%06d_sph_est.wav', f-1));
        if ~exist(estFile, 'file'), continue; end

        clean = mean(audioread(fullfile(cleanDir, files(f).name)), 2);
        noisy = mean(audioread(fullfile(noisyDir, files(f).name)), 2);
        denoised = mean(audioread(estFile), 2);

        minLen = min([length(clean), length(noisy), length(denoised)]);
        clean = clean(1:minLen); noisy = noisy(1:minLen); denoised = denoised(1:minLen);

        sisnri_list(end+1) = compute_sisnr(denoised, clean) - compute_sisnr(noisy, clean);
        sdri_list(end+1)   = compute_sdr(denoised, clean)   - compute_sdr(noisy, clean);
    end

    if ~isempty(sisnri_list)
        fprintf('\nConv-TasNet DeepShip 整体:\n');
        fprintf('  SI-SNRi=%.2f, SDRi=%.2f (n=%d)\n', mean(sisnri_list), mean(sdri_list), length(sisnri_list));
        fprintf('表4-1 DeepShip: SI-SNRi=%.2f, SDRi=%.2f\n', mean(sisnri_list), mean(sdri_list));
    end
end

function s = compute_sisnr(est, ref)
    est=est(:)-mean(est(:)); ref=ref(:)-mean(ref(:));
    s_target = (est'*ref)/(ref'*ref+1e-8)*ref; e=est-s_target;
    s=10*log10((s_target'*s_target)/(e'*e+1e-8)+1e-8);
end
function s = compute_sdr(est, ref)
    e=est(:)-ref(:); s=10*log10((ref(:)'*ref(:))/(e'*e+1e-8)+1e-8);
end