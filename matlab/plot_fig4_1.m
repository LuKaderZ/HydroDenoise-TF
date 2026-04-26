%% 图4-1：船舶信号、背景噪声及混合信号的PSD (1×4布局，黑体中文字体)
clear; clc; close all;

% ==================== 字体与样式设置 (中文黑体) ====================
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultTextFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultLineLineWidth', 1.0);

% ==================== 路径与参数配置 ====================
rawDataBase = 'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF\raw_data';
shipEarDir = fullfile(rawDataBase, 'ShipsEar');

passengerDir = fullfile(shipEarDir, 'passenger');
roroDir = fullfile(shipEarDir, 'roro');
noiseDir = fullfile(shipEarDir, 'wind');

passengerFiles = dir(fullfile(passengerDir, '*.wav'));
roroFiles = dir(fullfile(roroDir, '*.wav'));
noiseFiles = dir(fullfile(noiseDir, '*.wav'));

if isempty(passengerFiles) || isempty(roroFiles) || isempty(noiseFiles)
    error('缺少原始音频文件，请检查路径');
end

passengerPath = fullfile(passengerDir, passengerFiles(1).name);
roroPath = fullfile(roroDir, roroFiles(1).name);
noisePath = fullfile(noiseDir, noiseFiles(1).name);

fs_target = 16000;
duration = 3;
snr_mix = -15;

% ==================== 加载与预处理 ====================
[passenger, fs_orig] = audioread(passengerPath);
passenger = mean(passenger, 2);
if fs_orig ~= fs_target, passenger = resample(passenger, fs_target, fs_orig); end

[roro, fs_orig] = audioread(roroPath);
roro = mean(roro, 2);
if fs_orig ~= fs_target, roro = resample(roro, fs_target, fs_orig); end

[noise, fs_orig] = audioread(noisePath);
noise = mean(noise, 2);
if fs_orig ~= fs_target, noise = resample(noise, fs_target, fs_orig); end

N = fs_target * duration;
passenger = passenger(1:min(N, end));
roro = roro(1:min(N, end));
noise = noise(1:min(N, end));

noise_pass = prepare_noise(noise, length(passenger));
mix_passenger = mix_at_snr(passenger, noise_pass, snr_mix);
noise_roro = prepare_noise(noise, length(roro));
mix_roro = mix_at_snr(roro, noise_roro, snr_mix);

% ==================== 绘图参数 ====================
win_psd = hamming(1024);
noverlap_psd = 512;
nfft_psd = 1024;

% ==================== 绘图 ====================
figure('Units', 'centimeters', 'Position', [2, 2, 24, 7], 'Color', 'white');

% (a) 客船 PSD
subplot(1, 4, 1);
[pxx_pass, f] = pwelch(passenger, win_psd, noverlap_psd, nfft_psd, fs_target);
plot(f, 10*log10(pxx_pass), 'k', 'LineWidth', 0.8);
xlabel('频率 (Hz)'); ylabel('PSD (dB/Hz)');
title('(a) 客船 PSD', 'FontWeight', 'bold');
xlim([0 8000]); grid on; box on;
psd_data = 10*log10(pxx_pass);

% (b) 滚装船 PSD
subplot(1, 4, 2);
[pxx_roro, ~] = pwelch(roro, win_psd, noverlap_psd, nfft_psd, fs_target);
plot(f, 10*log10(pxx_roro), 'k', 'LineWidth', 0.8);
xlabel('频率 (Hz)'); ylabel('PSD (dB/Hz)');
title('(b) 滚装船 PSD', 'FontWeight', 'bold');
xlim([0 8000]); grid on; box on;
psd_data = [psd_data; 10*log10(pxx_roro)];

% (c) 背景噪声 PSD
subplot(1, 4, 3);
[pxx_noise, ~] = pwelch(noise, win_psd, noverlap_psd, nfft_psd, fs_target);
plot(f, 10*log10(pxx_noise), 'k', 'LineWidth', 0.8);
xlabel('频率 (Hz)'); ylabel('PSD (dB/Hz)');
title('(c) 背景噪声 PSD', 'FontWeight', 'bold');
xlim([0 8000]); grid on; box on;
psd_data = [psd_data; 10*log10(pxx_noise)];

% (d) 混合信号 PSD 对比
subplot(1, 4, 4);
[pxx_mp, ~] = pwelch(mix_passenger, win_psd, noverlap_psd, nfft_psd, fs_target);
[pxx_mr, ~] = pwelch(mix_roro, win_psd, noverlap_psd, nfft_psd, fs_target);
plot(f, 10*log10(pxx_mp), 'k-', f, 10*log10(pxx_mr), 'k--', 'LineWidth', 0.8);
xlabel('频率 (Hz)'); ylabel('PSD (dB/Hz)');
title(sprintf('(d) 混合信号 PSD (%d dB)', snr_mix), 'FontWeight', 'bold');
legend('客船混合', '滚装船混合', 'Location', 'southwest', 'Box', 'off');
xlim([0 8000]); grid on; box on;
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
saveDir = fullfile('C:\Users\XUWEILUN\Desktop\HydroDenoise-TF', 'figures');
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
pdfPath = fullfile(saveDir, 'fig4-1_ShipsEar_PSD.pdf');
pngPath = fullfile(saveDir, 'fig4-1_ShipsEar_PSD.png');
exportgraphics(gcf, pdfPath, 'ContentType', 'vector');
saveas(gcf, pngPath);
fprintf('图4-1已保存至 %s\n', saveDir);

% ==================== 辅助函数 ====================
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