%% 图4-1：船舶信号、背景噪声及混合信号的PSD与语谱图
clear; clc; close all;

% ==================== 路径与参数配置 ====================
rawDataBase = 'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF\raw_data';
shipEarDir = fullfile(rawDataBase, 'ShipsEar');

% 船舶信号文件 (请根据实际下载的文件名修改，这里自动选取第一个.wav文件)
passengerDir = fullfile(shipEarDir, 'passenger');
roroDir = fullfile(shipEarDir, 'roro');
noiseDir = fullfile(shipEarDir, 'wind');   % 或 'flow'

% 获取第一个wav文件作为示例
passengerFiles = dir(fullfile(passengerDir, '*.wav'));
roroFiles = dir(fullfile(roroDir, '*.wav'));
noiseFiles = dir(fullfile(noiseDir, '*.wav'));

if isempty(passengerFiles) || isempty(roroFiles) || isempty(noiseFiles)
    error('缺少原始音频文件，请检查路径');
end

passengerPath = fullfile(passengerDir, passengerFiles(1).name);
roroPath = fullfile(roroDir, roroFiles(1).name);
noisePath = fullfile(noiseDir, noiseFiles(1).name);

% 目标采样率
fs_target = 16000;
% 信号长度（秒），取前30秒用于绘图
duration = 30;

% 混合信噪比 (dB)
snr_mix = -15;

% ==================== 加载与预处理 ====================
[passenger, fs_orig] = audioread(passengerPath);
passenger = mean(passenger, 2);                      % 转单声道
if fs_orig ~= fs_target
    passenger = resample(passenger, fs_target, fs_orig);
end

[roro, fs_orig] = audioread(roroPath);
roro = mean(roro, 2);
if fs_orig ~= fs_target
    roro = resample(roro, fs_target, fs_orig);
end

[noise, fs_orig] = audioread(noisePath);
noise = mean(noise, 2);
if fs_orig ~= fs_target
    noise = resample(noise, fs_target, fs_orig);
end

% 截取前duration秒
N = fs_target * duration;
passenger = passenger(1:min(N, end));
roro = roro(1:min(N, end));
noise = noise(1:min(N, end));

% 合成混合信号 (用passenger作为目标)
% 对齐噪声长度
if length(noise) < length(passenger)
    rep = ceil(length(passenger)/length(noise));
    noise = repmat(noise, rep, 1);
end
noise = noise(1:length(passenger));

% 计算缩放因子
P_s = mean(passenger.^2);
P_n = mean(noise.^2);
alpha = sqrt(P_s / (P_n * 10^(snr_mix/10)));
mix = passenger + alpha * noise;

% ==================== 绘图 ====================
% 设置字体
set(0, 'DefaultAxesFontName', 'Microsoft YaHei');
set(0, 'DefaultTextFontName', 'Microsoft YaHei');
set(0, 'DefaultAxesFontSize', 10);

figure('Units', 'centimeters', 'Position', [2, 2, 18, 12], 'Color', 'white');

% --- 左列：PSD图 ---
% PSD参数
window = hamming(1024);
noverlap = 512;
nfft = 2048;

% 客船 PSD
subplot(2,3,1);
[pxx_pass, f] = pwelch(passenger, window, noverlap, nfft, fs_target);
plot(f, 10*log10(pxx_pass), 'k', 'LineWidth', 0.8);
xlabel('频率 (Hz)'); ylabel('功率谱密度 (dB/Hz)');
title('(a) 客船信号 PSD');
xlim([0 2000]); grid on; box on;

% 滚装船 PSD
subplot(2,3,2);
[pxx_roro, ~] = pwelch(roro, window, noverlap, nfft, fs_target);
plot(f, 10*log10(pxx_roro), 'k', 'LineWidth', 0.8);
xlabel('频率 (Hz)'); ylabel('功率谱密度 (dB/Hz)');
title('(b) 滚装船信号 PSD');
xlim([0 2000]); grid on; box on;

% 噪声 PSD
subplot(2,3,3);
[pxx_noise, ~] = pwelch(noise, window, noverlap, nfft, fs_target);
plot(f, 10*log10(pxx_noise), 'k', 'LineWidth', 0.8);
xlabel('频率 (Hz)'); ylabel('功率谱密度 (dB/Hz)');
title('(c) 背景噪声 PSD');
xlim([0 2000]); grid on; box on;

% --- 右列：语谱图 (时间-频率图) ---
% 语谱图参数
win = 256;
overlap = 200;

% 客船语谱图
subplot(2,3,4);
spectrogram(passenger, win, overlap, nfft, fs_target, 'yaxis');
title('(d) 客船信号语谱图');
ylim([0 2]); colorbar off;

% 滚装船语谱图
subplot(2,3,5);
spectrogram(roro, win, overlap, nfft, fs_target, 'yaxis');
title('(e) 滚装船信号语谱图');
ylim([0 2]); colorbar off;

% 混合信号语谱图 (-15 dB)
subplot(2,3,6);
spectrogram(mix, win, overlap, nfft, fs_target, 'yaxis');
title('(f) -15 dB 混合信号语谱图');
ylim([0 2]); colorbar off;

% 保存
saveDir = fullfile('C:\Users\XUWEILUN\Desktop\HydroDenoise-TF', 'figures');
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
pdfPath = fullfile(saveDir, 'fig4-1_ShipsEar_PSD_Spectrogram.pdf');
pngPath = fullfile(saveDir, 'fig4-1_ShipsEar_PSD_Spectrogram.png');
exportgraphics(gcf, pdfPath, 'ContentType', 'vector');
saveas(gcf, pngPath);
fprintf('图4-1已保存至 %s\n', saveDir);