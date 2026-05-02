%% 图4-4：各模型降噪后时频谱图对比
clear; clc; close all;

set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultTextFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 8);

% ==================== 路径 ====================
projectRoot = 'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF';
cleanDir = fullfile(projectRoot, 'data', 'ShipsEar', 'test1', 'clean');
noisyDir = fullfile(projectRoot, 'data', 'ShipsEar', 'test1', 'noisy');
crnEstDir = fullfile(projectRoot, 'baselines', 'CRN-causal', 'data', 'data', 'datasets', 'tt', 'tt_test1');
convtasnetEstDir = fullfile(projectRoot, 'experiments', 'conv_tasnet', 'estimates', 'tt_test1');
dprnnEstDir = fullfile(projectRoot, 'experiments', 'dprnn', 'estimates', 'tt_test1');
dcamfEstDir = fullfile(projectRoot, 'experiments', 'dcamf_net', 'denoised', 'ShipsEar_test1');

% ==================== 参数 ====================
fs = 16000;
nfft_spec = 512;
win_spec = hamming(256);
noverlap_spec = 200;
freqLim = [0, 4000];

% ==================== 选取与图4-2一致的样本 ====================
fprintf('遍历测试集，选取与图4-2一致的样本...\n');
cleanFiles = dir(fullfile(cleanDir, '*.wav'));
nSamples = length(cleanFiles);
bestSIi = -inf; bestIdx = 1;
analysisWin = 0.05;

for k = 1:nSamples
    fname = cleanFiles(k).name;
    dcamfFile = fullfile(dcamfEstDir, sprintf('%06d.wav', k));
    if ~exist(dcamfFile, 'file'), continue; end

    [clean, ~] = audioread(fullfile(cleanDir, fname)); clean = mean(clean,2);
    [noisy, ~] = audioread(fullfile(noisyDir, fname)); noisy = mean(noisy,2);
    dcamf = mean(audioread(dcamfFile), 2);

    minLen = min([length(clean), length(noisy), length(dcamf)]);
    clean=clean(1:minLen); noisy=noisy(1:minLen); dcamf=dcamf(1:minLen);
    winLen = round(analysisWin*fs);
    if minLen < winLen, continue; end
    nWindows = minLen - winLen + 1;
    energies = zeros(nWindows, 1);
    for s = 1:winLen:nWindows
        energies(s) = sum(noisy(s:s+winLen-1).^2);
    end
    [~, maxIdx] = max(energies);
    startS = max(1, maxIdx);
    endS = min(minLen, startS+winLen-1);
    if endS-startS+1 < winLen, continue; end
    sisnr_in = compute_sisnr(noisy(startS:endS), clean(startS:endS));
    sisnr_out = compute_sisnr(dcamf(startS:endS), clean(startS:endS));
    if sisnr_out - sisnr_in > bestSIi
        bestSIi = sisnr_out - sisnr_in;
        bestIdx = k;
    end
end
fprintf('选定样本: %d\n', bestIdx);

% ==================== 加载音频 ====================
fname = cleanFiles(bestIdx).name;
[clean, ~] = audioread(fullfile(cleanDir, fname)); clean = mean(clean,2);
[noisy, ~] = audioread(fullfile(noisyDir, fname)); noisy = mean(noisy,2);

crn = load_est(crnEstDir, bestIdx, 1);
ct  = load_est(convtasnetEstDir, bestIdx, 2);
dp  = load_est(dprnnEstDir, bestIdx, 2);
dc  = load_est(dcamfEstDir, bestIdx, 3);

minLen = min([length(clean), length(noisy), length(crn), length(ct), length(dp), length(dc)]);
clean=clean(1:minLen); noisy=noisy(1:minLen);
crn=crn(1:minLen); ct=ct(1:minLen); dp=dp(1:minLen); dc=dc(1:minLen);

% ==================== 计算语谱图 ====================
[~, F, T, p_clean] = spectrogram(clean, win_spec, noverlap_spec, nfft_spec, fs);
[~, ~, ~, p_noisy] = spectrogram(noisy, win_spec, noverlap_spec, nfft_spec, fs);
[~, ~, ~, p_crn]   = spectrogram(crn,   win_spec, noverlap_spec, nfft_spec, fs);
[~, ~, ~, p_ct]    = spectrogram(ct,    win_spec, noverlap_spec, nfft_spec, fs);
[~, ~, ~, p_dp]    = spectrogram(dp,    win_spec, noverlap_spec, nfft_spec, fs);
[~, ~, ~, p_dc]    = spectrogram(dc,    win_spec, noverlap_spec, nfft_spec, fs);

idxF = (F >= freqLim(1)) & (F <= freqLim(2));
F_plot = F(idxF);

specs = {p_clean, p_noisy, p_crn, p_ct, p_dp, p_dc};
titles = {'干净信号', '带噪信号', 'CRN', 'Conv-TasNet', 'DPRNN', 'DCAMF-Net'};
allPxx = [];
for i = 1:6, allPxx = [allPxx; 10*log10(specs{i}(idxF,:)+1e-10)]; end
cMin = floor(min(allPxx(:))/5)*5;
cMax = ceil(max(allPxx(:))/5)*5;

% ==================== 绘图 ====================
figure('Units', 'centimeters', 'Position', [2, 2, 24, 18], 'Color', 'white');

for i = 1:6
    subplot(3, 2, i);
    imagesc(T, F_plot/1000, 10*log10(specs{i}(idxF,:)+1e-10));
    axis xy; colormap(jet); caxis([cMin, cMax]);
    xlabel('时间 (s)'); ylabel('频率 (kHz)');
    title(titles{i}, 'FontWeight', 'bold');
    colorbar('Location', 'eastoutside', 'FontSize', 7);
end

% ==================== 保存 ====================
saveDir = fullfile(projectRoot, 'figures');
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
pdfPath = fullfile(saveDir, 'fig4-4_Spectrogram_Comparison.pdf');
pngPath = fullfile(saveDir, 'fig4-4_Spectrogram_Comparison.png');
exportgraphics(gcf, pdfPath, 'ContentType', 'vector');
saveas(gcf, pngPath);
fprintf('图4-4 语谱图对比已保存至 %s\n', saveDir);

% ==================== 函数 ====================
function si = compute_sisnr(est, ref)
    est = est(:) - mean(est(:));
    ref = ref(:) - mean(ref(:));
    dot_prod = sum(est .* ref);
    s_target = dot_prod * ref / (sum(ref.^2) + 1e-8);
    e_noise = est - s_target;
    si = 10 * log10(sum(s_target.^2) / (sum(e_noise.^2) + 1e-8) + 1e-8);
end

function est = load_est(estDir, idx, estType)
    if estType == 1
        f = fullfile(estDir, sprintf('%d_sph_est.wav', idx-1));
    elseif estType == 2
        f = fullfile(estDir, sprintf('%06d_sph_est.wav', idx-1));
    elseif estType == 3
        f = fullfile(estDir, sprintf('%06d.wav', idx));
    else, est = []; return;
    end
    if ~exist(f, 'file'), est = []; return; end
    est = mean(audioread(f), 2);
end
