%% 图4-1：船舶信号、背景噪声及混合信号的PSD (完整音频长度平均，黑体中文字体)
clear; clc; close all;

% ==================== 字体与样式设置 (中文黑体) ====================
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultTextFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultLineLineWidth', 1.0);

% ==================== 路径与参数配置 ====================
projectRoot = 'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF';
rawDataBase = fullfile(projectRoot, 'raw_data');
shipEarDir  = fullfile(rawDataBase, 'ShipsEar');

passengerDir = fullfile(shipEarDir, 'passenger');
roroDir      = fullfile(shipEarDir, 'roro');
noiseDir     = fullfile(shipEarDir, 'wind');

fs_target = 16000;
snr_mix   = -15;

% ==================== 绘图参数 ====================
win_psd      = hamming(1024);
noverlap_psd = 512;
nfft_psd     = 1024;

% ==================== 计算客船、滚装船、背景噪声的平均PSD（完整长度）====================
fprintf('计算客船平均PSD（完整长度）...\n');
avg_psd_passenger = compute_average_psd(passengerDir, fs_target, win_psd, noverlap_psd, nfft_psd);
fprintf('计算滚装船平均PSD（完整长度）...\n');
avg_psd_roro = compute_average_psd(roroDir, fs_target, win_psd, noverlap_psd, nfft_psd);
fprintf('计算背景噪声平均PSD（完整长度）...\n');
avg_psd_noise = compute_average_psd(noiseDir, fs_target, win_psd, noverlap_psd, nfft_psd);

% ==================== 准备混合信号（客船×1 + 滚装船×1 + 噪声×1）====================
passengerFiles = dir(fullfile(passengerDir, '*.wav'));
roroFiles      = dir(fullfile(roroDir, '*.wav'));
noiseFiles     = dir(fullfile(noiseDir, '*.wav'));

if isempty(passengerFiles) || isempty(roroFiles) || isempty(noiseFiles)
    error('缺少原始音频文件，请检查路径');
end

[passenger_sig, sr] = audioread(fullfile(passengerDir, passengerFiles(1).name));
passenger_sig = mean(passenger_sig, 2);
if sr ~= fs_target, passenger_sig = resample(passenger_sig, fs_target, sr); end

[roro_sig, sr] = audioread(fullfile(roroDir, roroFiles(1).name));
roro_sig = mean(roro_sig, 2);
if sr ~= fs_target, roro_sig = resample(roro_sig, fs_target, sr); end

[noise_sig, sr] = audioread(fullfile(noiseDir, noiseFiles(1).name));
noise_sig = mean(noise_sig, 2);
if sr ~= fs_target, noise_sig = resample(noise_sig, fs_target, sr); end

noise_pass = prepare_noise(noise_sig, length(passenger_sig));
mix_passenger = mix_at_snr(passenger_sig, noise_pass, snr_mix);
noise_roro = prepare_noise(noise_sig, length(roro_sig));
mix_roro = mix_at_snr(roro_sig, noise_roro, snr_mix);

fprintf('计算混合信号PSD...\n');
[pxx_mp, f] = pwelch(mix_passenger, win_psd, noverlap_psd, nfft_psd, fs_target);
[pxx_mr, ~] = pwelch(mix_roro, win_psd, noverlap_psd, nfft_psd, fs_target);

% ==================== 绘图 ====================
figure('Units', 'centimeters', 'Position', [2, 2, 24, 7], 'Color', 'white');

% (a) 客船 PSD
subplot(1, 4, 1);
plot(f, 10*log10(avg_psd_passenger), 'Color', [0.00 0.45 0.74], 'LineWidth', 1.0);
xlabel('频率 (Hz)'); ylabel('PSD (dB/Hz)');
title('(a) 客船 PSD', 'FontWeight', 'bold');
xlim([0 4000]); grid on; box on;
psd_data = 10*log10(avg_psd_passenger);

% (b) 滚装船 PSD
subplot(1, 4, 2);
plot(f, 10*log10(avg_psd_roro), 'Color', [0.47 0.67 0.19], 'LineWidth', 1.0);
xlabel('频率 (Hz)'); ylabel('PSD (dB/Hz)');
title('(b) 滚装船 PSD', 'FontWeight', 'bold');
xlim([0 4000]); grid on; box on;
psd_data = [psd_data; 10*log10(avg_psd_roro)];

% (c) 背景噪声 PSD
subplot(1, 4, 3);
plot(f, 10*log10(avg_psd_noise), 'Color', [0.50 0.50 0.50], 'LineWidth', 0.8);
xlabel('频率 (Hz)'); ylabel('PSD (dB/Hz)');
title('(c) 背景噪声 PSD', 'FontWeight', 'bold');
xlim([0 4000]); grid on; box on;
psd_data = [psd_data; 10*log10(avg_psd_noise)];

% (d) 混合信号 PSD 对比
subplot(1, 4, 4);
plot(f, 10*log10(pxx_mp), 'Color', [0.00 0.45 0.74], 'LineWidth', 1.0);
hold on;
plot(f, 10*log10(pxx_mr), 'Color', [0.47 0.67 0.19], 'LineWidth', 1.0);
xlabel('频率 (Hz)'); ylabel('PSD (dB/Hz)');
title(sprintf('(d) 混合信号 PSD (%d dB)', snr_mix), 'FontWeight', 'bold');
legend('客船混合', '滚装船混合', 'Location', 'southwest', 'Box', 'off');
xlim([0 4000]); grid on; box on;
psd_data = [psd_data; 10*log10(pxx_mp); 10*log10(pxx_mr)];

% --- 统一Y轴范围 ---
psd_min = min(psd_data(:));
psd_max = max(psd_data(:));
psd_margin = 0.05 * (psd_max - psd_min);
for k = 1:4
    subplot(1, 4, k);
    ylim([psd_min - psd_margin, psd_max + psd_margin]);
end

% ==================== 保存 ====================
saveDir = fullfile(projectRoot, 'figures');
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
pdfPath = fullfile(saveDir, 'fig4-1_ShipsEar_PSD.pdf');
pngPath = fullfile(saveDir, 'fig4-1_ShipsEar_PSD.png');
exportgraphics(gcf, pdfPath, 'ContentType', 'vector');
saveas(gcf, pngPath);
fprintf('图4-1 (基于完整长度平均PSD) 已保存至 %s\n', saveDir);

% ==================== 辅助函数 ====================
function avg_psd = compute_average_psd(folder, fs_target, window, noverlap, nfft)
files = dir(fullfile(folder, '*.wav'));
if isempty(files), error('文件夹 %s 中没有 wav 文件。', folder); end
avg_psd = [];
count   = 0;
for idx = 1:length(files)
    [sig, sr] = audioread(fullfile(folder, files(idx).name));
    sig = mean(sig, 2);
    if sr ~= fs_target, sig = resample(sig, fs_target, sr); end
    % 直接使用完整信号长度
    [pxx, ~] = pwelch(sig, window, noverlap, nfft, fs_target);
    if isempty(avg_psd)
        avg_psd = pxx;
    else
        avg_psd = avg_psd + pxx;
    end
    count = count + 1;
end
avg_psd = avg_psd / count;
end

function noise_out = prepare_noise(noise, target_len)
if length(noise) < target_len
    rep = ceil(target_len / length(noise));
    noise = repmat(noise, rep, 1);
end
start_idx = randi(length(noise) - target_len + 1);
noise_out = noise(start_idx : start_idx + target_len - 1);
end

function noisy = mix_at_snr(clean, noise, snr_db)
P_s = mean(clean.^2);
P_n = mean(noise.^2);
alpha = sqrt(P_s / (P_n * 10^(snr_db/10)));
noisy = clean + alpha * noise;
max_amp = max(abs(noisy));
if max_amp > 1.0, noisy = noisy / max_amp; end
end