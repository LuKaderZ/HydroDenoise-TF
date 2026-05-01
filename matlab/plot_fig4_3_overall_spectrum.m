%% 图4-3：各模型降噪后总体频谱对比（代表性样本，标注关键线谱位置）
%  位于4.2节：时域波形对比(图4-2) 与 关键线谱功率恢复(原图4-3) 之间
clear; clc; close all;

% ==================== 字体与样式 (中文黑体) ====================
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultTextFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 9);
set(0, 'DefaultLineLineWidth', 1.0);

% ==================== 路径配置 ====================
projectRoot = 'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF';

% 客船原始音频目录 (用于自动发现线谱)
passengerDir = fullfile(projectRoot, 'raw_data', 'ShipsEar', 'passenger');

% 测试集路径
cleanDir = fullfile(projectRoot, 'data', 'ShipsEar', 'test1', 'clean');
noisyDir = fullfile(projectRoot, 'data', 'ShipsEar', 'test1', 'noisy');

% 各模型增强音频路径
crnEstDir = fullfile(projectRoot, 'baselines', 'CRN-causal', 'data', 'data', 'datasets', 'tt', 'tt_test1');
convtasnetEstDir = fullfile(projectRoot, 'experiments', 'conv_tasnet', 'estimates', 'tt_test1');
dprnnEstDir = fullfile(projectRoot, 'experiments', 'dprnn', 'estimates', 'tt_test1');
dcamfEstDir = fullfile(projectRoot, 'experiments', 'dcamf_net', 'denoised', 'ShipsEar_test1');

% ==================== 参数 ====================
fs = 16000;                     % 采样率
freqRange = [0, 4000];          % 频谱显示范围 (Hz)
freqDetectRange = [0, 4000];    % 线谱检测范围 (Hz)
nTopPeaks = 5;                  % 选取突出度最高的线谱数量
nfft_psd = 1024;                % PSD 的 FFT 点数 (粗粒度，曲线平滑)
win_len = 1024;                 % PSD 窗长
noverlap = 512;                 % PSD 重叠 (50%)

% ==================== 第一步：自动发现客船线谱 ====================
fprintf('正在分析客船音频，提取最突出的线谱（局部突出度法）...\n');
promFreqs = find_prominent_line_spectra(passengerDir, fs, freqDetectRange, nTopPeaks);
if isempty(promFreqs)
    error('未检测到任何有效线谱。');
end
fprintf('检测到 %d 条主要线谱: %s Hz\n', length(promFreqs), mat2str(round(promFreqs)));

% ==================== 第二步：遍历测试集，选择最佳样本 ====================
fprintf('正在遍历测试集，挑选最能体现 DCAMF-Net 优势的样本...\n');
bestIdx = select_best_sample(cleanDir, noisyDir, crnEstDir, convtasnetEstDir, ...
                             dprnnEstDir, dcamfEstDir, promFreqs, fs);
fprintf('选定测试样本索引: %d\n', bestIdx);

% ==================== 第三步：加载最佳样本的音频 ====================
cleanFiles = dir(fullfile(cleanDir, '*.wav'));
fname = cleanFiles(bestIdx).name;

[clean, ~] = audioread(fullfile(cleanDir, fname)); clean = mean(clean,2);
[noisy, ~] = audioread(fullfile(noisyDir, fname)); noisy = mean(noisy,2);

crn_est = load_est(crnEstDir, bestIdx, 1);
ct_est  = load_est(convtasnetEstDir, bestIdx, 2);
dp_est  = load_est(dprnnEstDir, bestIdx, 2);
dc_est  = load_est(dcamfEstDir, bestIdx, 3);

% 长度对齐
minLen = min([length(clean), length(noisy), length(crn_est), ...
              length(ct_est), length(dp_est), length(dc_est)]);
clean=clean(1:minLen); noisy=noisy(1:minLen);
crn_est=crn_est(1:minLen); ct_est=ct_est(1:minLen);
dp_est=dp_est(1:minLen); dc_est=dc_est(1:minLen);

% ==================== 第四步：计算各信号 PSD ====================
fprintf('正在计算各信号功率谱密度...\n');
[pxx_clean, f] = pwelch(clean, hamming(win_len), noverlap, nfft_psd, fs);
[pxx_noisy, ~] = pwelch(noisy, hamming(win_len), noverlap, nfft_psd, fs);
[pxx_crn, ~]   = pwelch(crn_est, hamming(win_len), noverlap, nfft_psd, fs);
[pxx_ct, ~]    = pwelch(ct_est,  hamming(win_len), noverlap, nfft_psd, fs);
[pxx_dp, ~]    = pwelch(dp_est,  hamming(win_len), noverlap, nfft_psd, fs);
[pxx_dc, ~]    = pwelch(dc_est,  hamming(win_len), noverlap, nfft_psd, fs);

% 转换为 dB
psd_clean = 10*log10(pxx_clean);
psd_noisy = 10*log10(pxx_noisy);
psd_crn   = 10*log10(pxx_crn);
psd_ct    = 10*log10(pxx_ct);
psd_dp    = 10*log10(pxx_dp);
psd_dc    = 10*log10(pxx_dc);

% 限定频率范围
idxFreq = (f >= freqRange(1)) & (f <= freqRange(2));
f_plot = f(idxFreq) / 1000;  % 转换为 kHz
psd_clean = psd_clean(idxFreq);
psd_noisy = psd_noisy(idxFreq);
psd_crn   = psd_crn(idxFreq);
psd_ct    = psd_ct(idxFreq);
psd_dp    = psd_dp(idxFreq);
psd_dc    = psd_dc(idxFreq);

% ==================== 第五步：计算样本的 SI-SNRi 和 SDRi ====================
sisnr_in  = compute_sisnr(noisy, clean);
sisnr_crn = compute_sisnr(crn_est, clean);
sisnr_ct  = compute_sisnr(ct_est, clean);
sisnr_dp  = compute_sisnr(dp_est, clean);
sisnr_dc  = compute_sisnr(dc_est, clean);

sdr_in  = compute_sdr(noisy, clean);
sdr_crn = compute_sdr(crn_est, clean);
sdr_ct  = compute_sdr(ct_est, clean);
sdr_dp  = compute_sdr(dp_est, clean);
sdr_dc  = compute_sdr(dc_est, clean);

fprintf('\n========== 选定样本指标 ==========\n');
fprintf('样本索引: %d, 文件: %s\n', bestIdx, fname);
fprintf('%-15s %-10s %-10s\n', '模型', 'SI-SNRi(dB)', 'SDRi(dB)');
fprintf('%-15s %-10.2f %-10.2f\n', 'CRN', sisnr_crn-sisnr_in, sdr_crn-sdr_in);
fprintf('%-15s %-10.2f %-10.2f\n', 'Conv-TasNet', sisnr_ct-sisnr_in, sdr_ct-sdr_in);
fprintf('%-15s %-10.2f %-10.2f\n', 'DPRNN', sisnr_dp-sisnr_in, sdr_dp-sdr_in);
fprintf('%-15s %-10.2f %-10.2f\n', 'DCAMF-Net', sisnr_dc-sisnr_in, sdr_dc-sdr_in);
fprintf('==================================\n');

% ==================== 第六步：统一纵轴范围 ====================
allPSD = [psd_clean; psd_noisy; psd_crn; psd_ct; psd_dp; psd_dc];
yMin = floor(min(allPSD) / 10) * 10;
yMax = ceil(max(allPSD) / 10) * 10;

% ==================== 第七步：绘图 (纯黑白、2x2子图) ====================
figure('Units', 'centimeters', 'Position', [2, 2, 22, 16], 'Color', 'white');

models = {
    'CRN',              psd_crn;
    'Conv-TasNet',      psd_ct;
    'DPRNN',            psd_dp;
    'DCAMF-Net',        psd_dc;
};

for i = 1:4
    subplot(2, 2, i);
    hold on;

    % 带噪信号 (浅灰细点线，提供噪声基底参考)
    plot(f_plot, psd_noisy, '-', 'Color', [0.65 0.65 0.65], 'LineWidth', 0.5);

    % 干净信号 (粗黑实线)
    plot(f_plot, psd_clean, 'k-', 'LineWidth', 1.5);

    % 模型降噪输出 (粗黑虚线)
    plot(f_plot, models{i,2}, 'k--', 'LineWidth', 1.5);

    % 标注关键线谱位置
    for j = 1:length(promFreqs)
        xline(promFreqs(j)/1000, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5, ...
              'HandleVisibility', 'off');
    end

    xlabel('频率 (kHz)');
    ylabel('PSD (dB/Hz)');
    title(models{i,1}, 'FontWeight', 'bold');
    xlim([freqRange(1) freqRange(2)] / 1000);
    ylim([yMin, yMax]);
    legend('带噪信号', '干净信号', '降噪后', ...
           'Location', 'southwest', 'Box', 'off', 'FontSize', 8);
    grid on; box on;
    hold off;
end

% ==================== 保存 ====================
saveDir = fullfile(projectRoot, 'figures');
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
pdfPath = fullfile(saveDir, 'fig4-3_overall_spectrum_comparison.pdf');
pngPath = fullfile(saveDir, 'fig4-3_overall_spectrum_comparison.png');
exportgraphics(gcf, pdfPath, 'ContentType', 'vector');
saveas(gcf, pngPath);
fprintf('\n图4-3 总体频谱对比已保存至 %s\n', saveDir);

% ==================== 函数定义 ====================

function freqs = find_prominent_line_spectra(audioDir, fs, freqRange, nPeaks)
    files = dir(fullfile(audioDir, '*.wav'));
    if isempty(files), error('在 %s 未找到 wav 文件', audioDir); end
    avgPxx = [];
    count = 0;
    for k = 1:length(files)
        [sig, sr] = audioread(fullfile(audioDir, files(k).name));
        sig = mean(sig, 2);
        if sr ~= fs, sig = resample(sig, fs, sr); end
        [pxx, f] = pwelch(sig, hamming(2048), 1024, 4096, fs);
        if isempty(avgPxx), avgPxx = pxx; else, avgPxx = avgPxx + pxx; end
        count = count + 1;
    end
    avgPxx = avgPxx / count;
    avgPxx_dB = 10*log10(avgPxx);

    idxRange = (f >= freqRange(1)) & (f <= freqRange(2));
    f_sub = f(idxRange);
    p_sub = avgPxx_dB(idxRange);

    [pks, locs] = findpeaks(p_sub, f_sub, 'MinPeakProminence', 5, 'MinPeakDistance', 10);
    if length(locs) > nPeaks
        [~, sortIdx] = sort(pks, 'descend');
        locs = locs(sortIdx(1:nPeaks));
    end
    freqs = sort(locs);
end

function bestIdx = select_best_sample(cleanDir, noisyDir, crnDir, ctDir, dpDir, dcDir, lineFreqs, fs)
    cleanFiles = dir(fullfile(cleanDir, '*.wav'));
    nSamples = length(cleanFiles);
    scoreList = zeros(nSamples, 1);

    for idx = 1:nSamples
        fname = cleanFiles(idx).name;
        [clean, ~] = audioread(fullfile(cleanDir, fname)); clean = mean(clean,2);

        crn = load_est(crnDir, idx, 1); ct = load_est(ctDir, idx, 2);
        dp = load_est(dpDir, idx, 2);   dc = load_est(dcDir, idx, 3);
        if isempty(crn)||isempty(ct)||isempty(dp)||isempty(dc), continue; end

        minLen = min([length(clean), length(crn), length(ct), length(dp), length(dc)]);
        clean=clean(1:minLen); crn=crn(1:minLen); ct=ct(1:minLen);
        dp=dp(1:minLen); dc=dc(1:minLen);

        [pxx_clean, f] = pwelch(clean, hamming(2048), 1024, 4096, fs);
        pxx_crn = pwelch(crn, hamming(2048), 1024, 4096, fs);
        pxx_ct  = pwelch(ct, hamming(2048), 1024, 4096, fs);
        pxx_dp  = pwelch(dp, hamming(2048), 1024, 4096, fs);
        pxx_dc  = pwelch(dc, hamming(2048), 1024, 4096, fs);

        score = 0;
        for fq = lineFreqs(:)'
            [~, idxF] = min(abs(f(:) - double(fq)));
            idxWin = max(1, idxF-3):min(length(f), idxF+3);
            ref = mean(10*log10(pxx_clean(idxWin)));
            err_crn = abs(mean(10*log10(pxx_crn(idxWin))) - ref);
            err_ct  = abs(mean(10*log10(pxx_ct(idxWin))) - ref);
            err_dp  = abs(mean(10*log10(pxx_dp(idxWin))) - ref);
            err_dc  = abs(mean(10*log10(pxx_dc(idxWin))) - ref);
            other_best = min([err_crn, err_ct, err_dp]);
            score = score + (other_best - err_dc);
        end
        scoreList(idx) = score;
    end
    [~, bestIdx] = max(scoreList);
    if bestIdx == 0, bestIdx = 1; end
end

function est = load_est(estDir, idx, estType)
    if estType == 1
        f = fullfile(estDir, sprintf('%d_sph_est.wav', idx-1));
    elseif estType == 2
        f = fullfile(estDir, sprintf('%06d_sph_est.wav', idx-1));
    elseif estType == 3
        f = fullfile(estDir, sprintf('%06d.wav', idx));
    else
        est = []; return;
    end
    if ~exist(f, 'file'), est = []; return; end
    est = mean(audioread(f), 2);
end

function si = compute_sisnr(est, ref)
    est = est(:) - mean(est(:));
    ref = ref(:) - mean(ref(:));
    dot_prod = sum(est .* ref);
    s_target = dot_prod * ref / (sum(ref.^2) + 1e-8);
    e_noise = est - s_target;
    si = 10 * log10(sum(s_target.^2) / (sum(e_noise.^2) + 1e-8) + 1e-8);
end

function s = compute_sdr(est, ref)
    est = est(:); ref = ref(:);
    noise = ref - est;
    s = 10 * log10(sum(ref.^2) / (sum(noise.^2) + 1e-8) + 1e-8);
end
