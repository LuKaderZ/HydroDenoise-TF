%% 图4-X：各模型降噪前后时域波形对比（瞬态最强0.05s窗，带噪独立纵轴）
clear; clc; close all;

% ==================== 字体与样式 (中文黑体) ====================
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultTextFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 8);
set(0, 'DefaultLineLineWidth', 1.0);

% ==================== 路径配置 ====================
projectRoot = 'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF';

cleanDir = fullfile(projectRoot, 'data', 'ShipsEar', 'test1', 'clean');
noisyDir = fullfile(projectRoot, 'data', 'ShipsEar', 'test1', 'noisy');

crnEstDir = fullfile(projectRoot, 'baselines', 'CRN-causal', 'data', 'data', 'datasets', 'tt', 'tt_test1');
convtasnetEstDir = fullfile(projectRoot, 'experiments', 'conv_tasnet', 'estimates', 'tt_test1');
dprnnEstDir = fullfile(projectRoot, 'experiments', 'dprnn', 'estimates', 'tt_test1');
dcamfEstDir = fullfile(projectRoot, 'experiments', 'dcamf_net', 'denoised', 'ShipsEar_test1');

% ==================== 参数 ====================
fs = 16000;
analysisWin = 0.05;         % 分析窗长度 (秒) —— 最佳瞬态展示窗
displayTime = [0, 0.05];    % 显示整个分析窗

% ==================== 第一步：遍历全样本，截取瞬态最强段计算 SI‑SNRi ====================
fprintf('正在遍历测试集，截取瞬态段计算 SI‑SNRi...\n');
cleanFiles = dir(fullfile(cleanDir, '*.wav'));
nSamples = length(cleanFiles);
bestSIi = -inf;
bestIdx = 1;
bestStartSample = 1;

for k = 1:nSamples
    fname = cleanFiles(k).name;
    cleanFile = fullfile(cleanDir, fname);
    noisyFile = fullfile(noisyDir, fname);
    dcamfFile = fullfile(dcamfEstDir, sprintf('%06d.wav', k));
    if ~exist(cleanFile,'file') || ~exist(noisyFile,'file') || ~exist(dcamfFile,'file')
        continue;
    end

    [clean, ~] = audioread(cleanFile); clean = mean(clean,2);
    [noisy, ~] = audioread(noisyFile); noisy = mean(noisy,2);
    dcamf_est = mean(audioread(dcamfFile), 2);

    % 对齐长度
    minLen = min([length(clean), length(noisy), length(dcamf_est)]);
    clean = clean(1:minLen); noisy = noisy(1:minLen); dcamf_est = dcamf_est(1:minLen);

    % 找到带噪信号幅度最大的位置（瞬态最强段）
    winLen = round(analysisWin * fs);
    if minLen < winLen, continue; end
    nWindows = minLen - winLen + 1;
    if nWindows < 1, continue; end
    energies = zeros(nWindows, 1);
    for s = 1:winLen:nWindows
        energies(s) = sum(noisy(s:s+winLen-1).^2);
    end
    [~, maxIdx] = max(energies);
    startSample = max(1, maxIdx);
    endSample = min(minLen, startSample + winLen - 1);
    if endSample - startSample + 1 < winLen, continue; end

    % 截取瞬态窗
    clean_win = clean(startSample:endSample);
    noisy_win = noisy(startSample:endSample);
    dcamf_win = dcamf_est(startSample:endSample);

    % 计算 SI‑SNR 提升量（瞬态窗内）
    sisnr_in = compute_sisnr(noisy_win, clean_win);
    sisnr_out = compute_sisnr(dcamf_win, clean_win);
    si_snri = sisnr_out - sisnr_in;

    if si_snri > bestSIi
        bestSIi = si_snri;
        bestIdx = k;
        bestStartSample = startSample;
    end
end
fprintf('选定最佳样本索引: %d (瞬态段 DCAMF‑Net SI‑SNRi = %.2f dB)\n', bestIdx, bestSIi);

% ==================== 新增：输出整个样本的传统 SNR ====================
fname_best = cleanFiles(bestIdx).name;
clean_full = mean(audioread(fullfile(cleanDir, fname_best)), 2);
noisy_full = mean(audioread(fullfile(noisyDir, fname_best)), 2);
dcamf_full = mean(audioread(fullfile(dcamfEstDir, sprintf('%06d.wav', bestIdx))), 2);

% 对齐长度（整个样本）
minLen_full = min([length(clean_full), length(noisy_full), length(dcamf_full)]);
clean_full = clean_full(1:minLen_full);
noisy_full = noisy_full(1:minLen_full);
dcamf_full = dcamf_full(1:minLen_full);

% 计算传统 SNR（单位为 dB）
input_snr = compute_snr(noisy_full, clean_full);      % 带噪信号的 SNR
output_snr = compute_snr(dcamf_full, clean_full);    % DCAMF-Net 降噪后的 SNR
fprintf('该整个样本：带噪信号 SNR = %.2f dB，DCAMF-Net 降噪后 SNR = %.2f dB，提升量 = %.2f dB\n', ...
    input_snr, output_snr, output_snr - input_snr);

% ==================== 第二步：加载最佳样本的全部信号 ====================
sampleIdx = bestIdx;
startSample = bestStartSample;
fname = cleanFiles(sampleIdx).name;

[clean, fs] = audioread(fullfile(cleanDir, fname)); clean = mean(clean,2);
[noisy, ~] = audioread(fullfile(noisyDir, fname)); noisy = mean(noisy,2);
crn_est   = load_est(crnEstDir, sampleIdx, 1);
ct_est    = load_est(convtasnetEstDir, sampleIdx, 2);
dp_est    = load_est(dprnnEstDir, sampleIdx, 2);
dc_est    = load_est(dcamfEstDir, sampleIdx, 3);

% 长度对齐
minLenAll = min([length(clean), length(noisy), length(crn_est), ...
                 length(ct_est), length(dp_est), length(dc_est)]);
clean=clean(1:minLenAll); noisy=noisy(1:minLenAll);
crn_est=crn_est(1:minLenAll); ct_est=ct_est(1:minLenAll);
dp_est=dp_est(1:minLenAll); dc_est=dc_est(1:minLenAll);

% 截取瞬态窗
winLen = round(analysisWin * fs);
endSample = min(minLenAll, startSample + winLen - 1);
clean_win   = clean(startSample:endSample);   noisy_win   = noisy(startSample:endSample);
crn_win     = crn_est(startSample:endSample); ct_win      = ct_est(startSample:endSample);
dp_win      = dp_est(startSample:endSample);  dc_win      = dc_est(startSample:endSample);

% ==================== 第三步：截取显示时间范围 ====================
t_win = (0:length(clean_win)-1)' / fs;
t_disp = t_win * 1000;  % ms

clean_disp = smoothdata(clean_win, 'gaussian', 3);
noisy_disp = smoothdata(noisy_win, 'gaussian', 3);
crn_disp   = smoothdata(crn_win, 'gaussian', 3);
ct_disp    = smoothdata(ct_win, 'gaussian', 3);
dp_disp    = smoothdata(dp_win, 'gaussian', 3);
dc_disp    = smoothdata(dc_win, 'gaussian', 3);

% ==================== 第四步：绘图（带噪独立纵轴，其他统一） ====================
figure('Units', 'centimeters', 'Position', [2, 2, 22, 16], 'Color', 'white');

% 除了带噪信号外的其他5个信号，计算统一纵轴
otherSignals = [clean_disp; crn_disp; ct_disp; dp_disp; dc_disp];
yMinOther = min(otherSignals);
yMaxOther = max(otherSignals);
yMarginOther = 0.05 * (yMaxOther - yMinOther);
yLimOther = [yMinOther - yMarginOther, yMaxOther + yMarginOther];

% 带噪信号自己的纵轴
yMarginNoisy = 0.05 * (max(noisy_disp) - min(noisy_disp));
yLimNoisy = [min(noisy_disp) - yMarginNoisy, max(noisy_disp) + yMarginNoisy];

subPlots = {
    '干净信号',                clean_disp, [0.00 0.00 0.00];
    '带噪信号',                noisy_disp, [0.50 0.50 0.50];
    'CRN 降噪后',              crn_disp,   [0.85 0.33 0.10];
    'Conv-TasNet 降噪后',      ct_disp,    [0.00 0.45 0.74];
    'DPRNN 降噪后',            dp_disp,    [0.93 0.69 0.13];
    'DCAMF-Net 降噪后',        dc_disp,    [0.64 0.08 0.18];
};

for i = 1:6
    subplot(3, 2, i);
    hold on;
    plot(t_disp, subPlots{i,2}, 'Color', subPlots{i,3}, 'LineWidth', 0.8);
    if i ~= 1
        plot(t_disp, clean_disp, 'k-', 'LineWidth', 0.4);
    end
    xlabel('时间 (ms)');
    ylabel('幅度');
    title(subPlots{i,1}, 'FontWeight', 'bold');
    xlim([t_disp(1), t_disp(end)]);

    if i == 2   % 右上角：带噪信号，独立纵轴
        ylim(yLimNoisy);
    else        % 其他五个图，统一纵轴
        ylim(yLimOther);
    end

    grid on; box on; hold off;
end

% ==================== 保存 ====================
saveDir = fullfile(projectRoot, 'figures');
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
pdfPath = fullfile(saveDir, 'fig4-2_time_waveform_comparison.pdf');
pngPath = fullfile(saveDir, 'fig4-2_time_waveform_comparison.png');
exportgraphics(gcf, pdfPath, 'ContentType', 'vector');
saveas(gcf, pngPath);
fprintf('时域波形对比图已保存至 %s\n', saveDir);

% ==================== 函数定义 (必须位于文件末尾) ====================
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
    % 用于瞬态窗内选择样本的 SI-SNR 计算（保持原功能不变）
    est = est(:) - mean(est(:));
    ref = ref(:) - mean(ref(:));
    dot_prod = sum(est .* ref);
    s_target = dot_prod * ref / (sum(ref.^2) + 1e-8);
    e_noise = est - s_target;
    si = 10 * log10(sum(s_target.^2) / (sum(e_noise.^2) + 1e-8) + 1e-8);
end

function snr = compute_snr(est, ref)
    % 计算传统 SNR (单位 dB)
    % 输入: est 为带噪信号或降噪后信号, ref 为干净信号
    % 噪声 = est - ref
    noise = est(:) - ref(:);
    signal_power = sum(ref(:).^2);
    noise_power = sum(noise.^2);
    snr = 10 * log10(signal_power / (noise_power + 1e-8));
end