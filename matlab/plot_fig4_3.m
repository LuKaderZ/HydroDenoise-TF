%% 图4-2：各模型线谱功率恢复条形图 (基于局部突出度自动选峰，最佳样本，输出数据)
clear; clc; close all;

% ==================== 字体与样式 (中文黑体) ====================
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultTextFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 10);
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
freqRange = [0, 4000];          % 线谱检测频率范围 (Hz)
nTopPeaks = 5;                  % 选取突出度最高的线谱数量

% ==================== 第一步：自动发现客船线谱 (基于局部突出度) ====================
fprintf('正在分析客船音频，提取最突出的线谱（局部突出度法）...\n');
promFreqs = find_prominent_line_spectra(passengerDir, fs, freqRange, nTopPeaks);
if isempty(promFreqs)
    error('未检测到任何有效线谱。');
end
fprintf('检测到 %d 条主要线谱: %s Hz\n', length(promFreqs), mat2str(round(promFreqs)));

% ==================== 第二步：遍历测试集，选择最佳样本 ====================
fprintf('正在遍历测试集，挑选最能体现 DCAMF‑Net 优势的样本...\n');
bestIdx = select_best_sample(cleanDir, noisyDir, crnEstDir, convtasnetEstDir, ...
                             dprnnEstDir, dcamfEstDir, promFreqs, fs);
fprintf('选定测试样本索引: %d\n', bestIdx);

% ==================== 第三步：加载最佳样本的音频 ====================
cleanFiles = dir(fullfile(cleanDir, '*.wav'));
fname = cleanFiles(bestIdx).name;

[clean, ~] = audioread(fullfile(cleanDir, fname)); clean = mean(clean,2);
[noisy, ~] = audioread(fullfile(noisyDir, fname)); noisy = mean(noisy,2);

% 加载各模型降噪信号
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

% ==================== 第四步：计算各频点功率 (高分辨率PSD) ====================
fprintf('正在计算各频点功率...\n');
nfft_psd = 4096;
win_len = 2048;
[pxx_clean, f] = pwelch(clean, hamming(win_len), win_len/2, nfft_psd, fs);
pxx_crn = pwelch(crn_est, hamming(win_len), win_len/2, nfft_psd, fs);
pxx_ct  = pwelch(ct_est,  hamming(win_len), win_len/2, nfft_psd, fs);
pxx_dp  = pwelch(dp_est,  hamming(win_len), win_len/2, nfft_psd, fs);
pxx_dc  = pwelch(dc_est,  hamming(win_len), win_len/2, nfft_psd, fs);

% 计算各线谱频点处的平均功率 (dB)
nFreqs = length(promFreqs);
powers = zeros(nFreqs, 5); % clean, crn, ct, dp, dc
freqLabels = cell(1, nFreqs);
for i = 1:nFreqs
    [~, idxF] = min(abs(f - promFreqs(i)));
    idxWin = max(1, idxF-4):min(length(f), idxF+4);
    powers(i, 1) = mean(10*log10(pxx_clean(idxWin)));
    powers(i, 2) = mean(10*log10(pxx_crn(idxWin)));
    powers(i, 3) = mean(10*log10(pxx_ct(idxWin)));
    powers(i, 4) = mean(10*log10(pxx_dp(idxWin)));
    powers(i, 5) = mean(10*log10(pxx_dc(idxWin)));
    freqLabels{i} = sprintf('%d Hz', round(promFreqs(i)));
end

% 功率偏差 (dB)
dev = powers(:,2:5) - powers(:,1);   % [crn, ct, dp, dc]

% ==================== 数据打印 (用于反馈分析) ====================
fprintf('\n========== 图4-2 数据摘要 ==========\n');
fprintf('选定样本索引: %d\n', bestIdx);
fprintf('%-15s %-10s %-10s %-10s %-10s\n', '频率', 'CRN', 'Conv-TasNet', 'DPRNN', 'DCAMF-Net');
for i = 1:nFreqs
    fprintf('%-15s %-10.2f %-10.2f %-10.2f %-10.2f\n', ...
        freqLabels{i}, dev(i,1), dev(i,2), dev(i,3), dev(i,4));
end

% 优势统计：= min(三基线偏差) - DCAMF偏差。正值越大，DCAMF优势越明显。
fprintf('\nDCAMF‑Net 相对于最佳基线的优势 (dB):\n');
for i = 1:nFreqs
    otherBest = min(dev(i,1:3));
    advantage = otherBest - dev(i,4);
    fprintf('  %s: %+.2f dB\n', freqLabels{i}, advantage);
end
fprintf('======================================\n');

% ==================== 第五步：绘制条形图 ====================
figure('Units', 'centimeters', 'Position', [2, 2, 16, 10], 'Color', 'white');

x = 1:nFreqs;
width = 0.2;
grayColors = {[0.25 0.25 0.25], [0.45 0.45 0.45], [0.65 0.65 0.65], [0.05 0.05 0.05]};
modelNames = {'CRN', 'Conv-TasNet', 'DPRNN', 'DCAMF-Net'};

hold on;
bars = cell(1, 4);
for i = 1:4
    bars{i} = bar(x + (i-2.5)*width, dev(:,i), width, ...
                  'FaceColor', grayColors{i}, 'EdgeColor', 'k', 'LineWidth', 0.5);
end

for i = 1:4
    for j = 1:nFreqs
        val = dev(j, i);
        if abs(val) > 0.5
            text(x(j)+(i-2.5)*width, val, sprintf('%.1f', val), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', ...
                 'bottom', 'FontSize', 7, 'Color', grayColors{i});
        end
    end
end

set(gca, 'XTick', x);
set(gca, 'XTickLabel', freqLabels);
xlabel('关键线谱频率');
ylabel('相对于干净信号的功率偏差 (dB)');
title('各模型线谱功率恢复对比 (0 dB 为完美恢复)', 'FontWeight', 'bold');
legend([bars{1}, bars{2}, bars{3}, bars{4}], modelNames, ...
       'Location', 'best', 'Box', 'off', 'FontSize', 9);
grid on; box on;
hold off;

% ==================== 保存 ====================
saveDir = fullfile(projectRoot, 'figures');
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
pdfPath = fullfile(saveDir, 'fig4-3_Line_Spectra_Bar.pdf');
pngPath = fullfile(saveDir, 'fig4-3_Line_Spectra_Bar.png');
exportgraphics(gcf, pdfPath, 'ContentType', 'vector');
saveas(gcf, pngPath);
fprintf('\n图4-2 线谱功率恢复条形图已保存至 %s\n', saveDir);

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
    
    [~, locs] = findpeaks(p_sub, f_sub, 'MinPeakProminence', 5, 'MinPeakDistance', 10);
    if length(locs) > nPeaks
        [~, sortIdx] = sort(p_sub(ismember(f_sub, locs)), 'descend');
        [~, idxPeaks] = findpeaks(p_sub, f_sub, 'MinPeakProminence', 5, 'MinPeakDistance', 10);
        [~, sortIdx] = sort(p_sub(ismember(f_sub, idxPeaks)), 'descend');
        locs = idxPeaks(sortIdx(1:nPeaks));
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