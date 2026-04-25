%% CRN 基线模型评估脚本（仅 test1，仅打印指标）
clear; clc;

% ==================== 路径配置 ====================
dataBasePath = 'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF\data';
crnEstBasePath = 'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF\baselines\CRN-causal\data\data\datasets\tt';

testName = 'test1';
targetSNRs = [-15, -10, -5];
tolerance = 2.5;

% ==================== 处理 ShipsEar test1 ====================
cleanDir = fullfile(dataBasePath, 'ShipsEar', testName, 'clean');
noisyDir = fullfile(dataBasePath, 'ShipsEar', testName, 'noisy');
denoisedDir = fullfile(crnEstBasePath, ['tt_' testName]);

if ~exist(denoisedDir, 'dir')
    error('增强音频目录 %s 不存在。', denoisedDir);
end

wavFiles = dir(fullfile(cleanDir, '*.wav'));
if isempty(wavFiles)
    wavFiles = dir(fullfile(cleanDir, '*.flac'));
end
[~, idx] = sort({wavFiles.name});
wavFiles = wavFiles(idx);

sisnri_cell = cell(1, length(targetSNRs));
sdri_cell   = cell(1, length(targetSNRs));
for s = 1:length(targetSNRs)
    sisnri_cell{s} = [];
    sdri_cell{s}   = [];
end

fprintf('\n========== 处理 ShipsEar %s ==========\n', testName);

for f = 1:length(wavFiles)
    fname = wavFiles(f).name;
    cleanPath = fullfile(cleanDir, fname);
    noisyPath = fullfile(noisyDir, fname);
    estName = sprintf('%d_sph_est.wav', f-1);
    denoisedPath = fullfile(denoisedDir, estName);

    if ~exist(denoisedPath, 'file')
        continue;
    end

    [clean, ~] = audioread(cleanPath);
    noisy = audioread(noisyPath);
    denoised = audioread(denoisedPath);

    if size(clean,2) > 1, clean = mean(clean,2); end
    if size(noisy,2) > 1, noisy = mean(noisy,2); end
    if size(denoised,2) > 1, denoised = mean(denoised,2); end

    minLen = min([length(clean), length(noisy), length(denoised)]);
    clean = clean(1:minLen);
    noisy = noisy(1:minLen);
    denoised = denoised(1:minLen);

    noise_actual = noisy - clean;
    P_s = mean(clean.^2);
    P_n = mean(noise_actual.^2);
    actualSNR = 10 * log10(P_s / (P_n + 1e-10));

    [~, snrIdx] = min(abs(targetSNRs - actualSNR));
    if abs(targetSNRs(snrIdx) - actualSNR) > tolerance
        continue;
    end

    sisnr_in = compute_sisnr(noisy, clean);
    sisnr_out = compute_sisnr(denoised, clean);
    sdr_in = compute_sdr(noisy, clean);
    sdr_out = compute_sdr(denoised, clean);

    sisnri_cell{snrIdx}(end+1) = sisnr_out - sisnr_in;
    sdri_cell{snrIdx}(end+1)   = sdr_out - sdr_in;
end

% ==================== 输出结果 ====================
fprintf('\n========== CRN test1 指标 ==========\n');
for s = 1:length(targetSNRs)
    if ~isempty(sisnri_cell{s})
        fprintf('  SNR ~%d dB: SI-SNRi = %.2f dB, SDRi = %.2f dB (样本数: %d)\n', ...
            targetSNRs(s), mean(sisnri_cell{s}), mean(sdri_cell{s}), length(sisnri_cell{s}));
    end
end

fprintf('\n填入表4-1：\n');
fprintf('  SI-SNRi = %.2f dB\n', mean(cell2mat(sisnri_cell)));
fprintf('  SDRi    = %.2f dB\n', mean(cell2mat(sdri_cell)));

% ==================== 辅助函数 ====================
function s = compute_sisnr(est, ref)
    est = est(:) - mean(est(:));
    ref = ref(:) - mean(ref(:));
    dot_prod = sum(est .* ref);
    s_target = dot_prod * ref / (sum(ref.^2) + 1e-8);
    e_noise = est - s_target;
    s = 10 * log10(sum(s_target.^2) / (sum(e_noise.^2) + 1e-8) + 1e-8);
end

function s = compute_sdr(est, ref)
    est = est(:); ref = ref(:);
    noise = ref - est;
    s = 10 * log10(sum(ref.^2) / (sum(noise.^2) + 1e-8) + 1e-8);
end