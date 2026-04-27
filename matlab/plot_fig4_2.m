%% 图4-2：各模型恢复信号平均功率谱密度对比 (遍历所有样本)
clear; clc; close all;

% ==================== 字体与样式 (中文黑体) ====================
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultTextFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 9);
set(0, 'DefaultLineLineWidth', 1.0);

% ==================== 路径配置 ====================
projectRoot = 'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF';
cleanDir = fullfile(projectRoot, 'data', 'ShipsEar', 'test1', 'clean');
crnEstDir = fullfile(projectRoot, 'baselines', 'CRN-causal', 'data', 'data', 'datasets', 'tt', 'tt_test1');
convtasnetEstDir = fullfile(projectRoot, 'experiments', 'conv_tasnet', 'estimates', 'tt_test1');
dprnnEstDir = fullfile(projectRoot, 'experiments', 'dprnn', 'estimates', 'tt_test1');
dcamfEstDir = fullfile(projectRoot, 'experiments', 'dcamf_net', 'denoised', 'ShipsEar_test1');

% ==================== 参数 ====================
fs = 16000;
window = hamming(1024); noverlap = 512; nfft = 1024;

% ==================== 初始化累加器 ====================
cleanFiles = dir(fullfile(cleanDir, '*.wav'));
nSamples = length(cleanFiles);

% 初始化平均PSD
avg_psd_clean = [];
avg_psd_crn = [];
avg_psd_convtasnet = [];
avg_psd_dprnn = [];
avg_psd_dcamf = [];
validCount = 0;

fprintf('正在遍历 %d 个样本计算平均PSD...\n', nSamples);

for k = 1:nSamples
    % 干净信号
    [clean, ~] = audioread(fullfile(cleanDir, cleanFiles(k).name));
    clean = mean(clean, 2);
    
    % CRN (文件名: (k-1)_sph_est.wav)
    crnFile = fullfile(crnEstDir, sprintf('%d_sph_est.wav', k-1));
    if ~exist(crnFile, 'file'), continue; end
    crn_est = mean(audioread(crnFile), 2);
    
    % Conv-TasNet
    convtasnetFile = fullfile(convtasnetEstDir, sprintf('%06d_sph_est.wav', k-1));
    if ~exist(convtasnetFile, 'file'), continue; end
    convtasnet_est = mean(audioread(convtasnetFile), 2);
    
    % DPRNN
    dprnnFile = fullfile(dprnnEstDir, sprintf('%06d_sph_est.wav', k-1));
    if ~exist(dprnnFile, 'file'), continue; end
    dprnn_est = mean(audioread(dprnnFile), 2);
    
    % DCAMF-Net
    dcamfFile = fullfile(dcamfEstDir, sprintf('%06d.wav', k));
    if ~exist(dcamfFile, 'file'), continue; end
    dcamf_est = mean(audioread(dcamfFile), 2);
    
    % 长度对齐
    minLen = min([length(clean), length(crn_est), length(convtasnet_est), ...
                  length(dprnn_est), length(dcamf_est)]);
    clean = clean(1:minLen);
    crn_est = crn_est(1:minLen);
    convtasnet_est = convtasnet_est(1:minLen);
    dprnn_est = dprnn_est(1:minLen);
    dcamf_est = dcamf_est(1:minLen);
    
    % 计算PSD
    [pxx_clean, f] = pwelch(clean, window, noverlap, nfft, fs);
    [pxx_crn, ~] = pwelch(crn_est, window, noverlap, nfft, fs);
    [pxx_convtasnet, ~] = pwelch(convtasnet_est, window, noverlap, nfft, fs);
    [pxx_dprnn, ~] = pwelch(dprnn_est, window, noverlap, nfft, fs);
    [pxx_dcamf, ~] = pwelch(dcamf_est, window, noverlap, nfft, fs);
    
    % 累加 (线性域累加，最后再取对数)
    if isempty(avg_psd_clean)
        avg_psd_clean = pxx_clean;
        avg_psd_crn = pxx_crn;
        avg_psd_convtasnet = pxx_convtasnet;
        avg_psd_dprnn = pxx_dprnn;
        avg_psd_dcamf = pxx_dcamf;
    else
        avg_psd_clean = avg_psd_clean + pxx_clean;
        avg_psd_crn = avg_psd_crn + pxx_crn;
        avg_psd_convtasnet = avg_psd_convtasnet + pxx_convtasnet;
        avg_psd_dprnn = avg_psd_dprnn + pxx_dprnn;
        avg_psd_dcamf = avg_psd_dcamf + pxx_dcamf;
    end
    validCount = validCount + 1;
    
    if mod(k, 500) == 0
        fprintf('  已处理 %d/%d 个样本\n', k, nSamples);
    end
end

% 取平均并转换到dB
avg_psd_clean       = 10*log10(avg_psd_clean / validCount);
avg_psd_crn         = 10*log10(avg_psd_crn / validCount);
avg_psd_convtasnet  = 10*log10(avg_psd_convtasnet / validCount);
avg_psd_dprnn       = 10*log10(avg_psd_dprnn / validCount);
avg_psd_dcamf       = 10*log10(avg_psd_dcamf / validCount);

fprintf('平均PSD计算完成，有效样本数: %d\n', validCount);

% ==================== 统一坐标轴范围 ====================
allPSD = [avg_psd_clean; avg_psd_crn; avg_psd_convtasnet; avg_psd_dprnn; avg_psd_dcamf];
yMin = floor(min(allPSD) / 10) * 10;
yMax = ceil(max(allPSD) / 10) * 10;
xLimits = [0 8];  % kHz

% ==================== 绘图 ====================
figure('Units', 'centimeters', 'Position', [2, 2, 18, 14], 'Color', 'white');

models = {
    'CRN',              avg_psd_crn,         'CRN估计';
    'Conv-TasNet',      avg_psd_convtasnet,  'Conv-TasNet估计';
    'DPRNN',            avg_psd_dprnn,       'DPRNN估计';
    'DCAMF-Net',        avg_psd_dcamf,       'DCAMF-Net估计';
};

for i = 1:4
    subplot(2, 2, i);
    hold on;
    
    % 干净信号（粗黑实线）
    plot(f/1000, avg_psd_clean, 'k-', 'LineWidth', 1.5);
    
    % 当前模型的估计信号（虚线）
    plot(f/1000, models{i,2}, 'k--', 'LineWidth', 1.2);
    
    xlabel('频率 (kHz)');
    ylabel('PSD (dB/Hz)');
    title(models{i,1}, 'FontWeight', 'bold');
    
    xlim(xLimits);
    ylim([yMin, yMax]);
    grid on; box on;
    
    legend('干净信号', models{i,3}, 'Location', 'southwest', 'Box', 'off', 'FontSize', 8);
    hold off;
end

% ==================== 保存 ====================
saveDir = fullfile(projectRoot, 'figures');
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
pdfPath = fullfile(saveDir, 'fig4-2_PSD_Model_Comparison.pdf');
pngPath = fullfile(saveDir, 'fig4-2_PSD_Model_Comparison.png');
exportgraphics(gcf, pdfPath, 'ContentType', 'vector');
saveas(gcf, pngPath);
fprintf('图4-2 (基于%d个样本的平均PSD) 已保存至 %s\n', validCount, saveDir);