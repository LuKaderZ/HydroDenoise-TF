%% 图4-5：DCAMF-Net泛化性能评估
clear; clc; close all;

% ==================== 字体设置（支持中文） ====================
set(0, 'DefaultAxesFontName', 'Microsoft YaHei');
set(0, 'DefaultTextFontName', 'Microsoft YaHei');
set(0, 'DefaultAxesFontSize', 11);

% ==================== 路径配置 ====================
projectRoot = 'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF';
dataBase = fullfile(projectRoot, 'data');
denoisedBase = fullfile(projectRoot, 'experiments', 'dcamf_net', 'denoised');
figuresDir = fullfile(projectRoot, 'figures');
if ~exist(figuresDir, 'dir'), mkdir(figuresDir); end

% ==================== 参数 ====================
targetSNRs = [-15, -10, -5];

testSets = {
    'ShipsEar/test1', 'ShipsEar_test1', '测试集一 (已知船型+已知噪声)';
    'ShipsEar/test2', 'ShipsEar_test2', '测试集二 (未知船型+已知噪声)';
    'ShipsEar/test3', 'ShipsEar_test3', '测试集三 (已知船型+未知噪声)';
};

numSets = size(testSets, 1);

% ==================== 计算每个样本的提升量 ====================
allResults = [];

for s = 1:numSets
    dataSubpath = testSets{s, 1};
    denoisedSubname = testSets{s, 2};
    
    cleanDir = fullfile(dataBase, dataSubpath, 'clean');
    noisyDir = fullfile(dataBase, dataSubpath, 'noisy');
    denoisedDir = fullfile(denoisedBase, denoisedSubname);
    
    if ~exist(denoisedDir, 'dir')
        fprintf('[跳过] %s\n', denoisedDir);
        continue;
    end
    
    files = dir(fullfile(cleanDir, '*.wav'));
    if isempty(files), files = dir(fullfile(cleanDir, '*.flac')); end
    [~, idx] = sort({files.name});
    files = files(idx);
    
    fprintf('[处理] %s (%d 文件)\n', dataSubpath, length(files));
    
    for f = 1:length(files)
        fname = files(f).name;
        cleanPath = fullfile(cleanDir, fname);
        noisyPath = fullfile(noisyDir, fname);
        denoisedPath = fullfile(denoisedDir, fname);
        
        if ~exist(denoisedPath, 'file'), continue; end
        
        [clean, ~] = audioread(cleanPath);
        noisy = audioread(noisyPath);
        denoised = audioread(denoisedPath);
        
        if size(clean,2)>1, clean = mean(clean,2); end
        if size(noisy,2)>1, noisy = mean(noisy,2); end
        if size(denoised,2)>1, denoised = mean(denoised,2); end
        
        minLen = min([length(clean), length(noisy), length(denoised)]);
        clean = clean(1:minLen); noisy = noisy(1:minLen); denoised = denoised(1:minLen);
        
        noiseActual = noisy - clean;
        actualSNR = 10 * log10(mean(clean.^2) / (mean(noiseActual.^2) + 1e-10));
        [~, snrIdx] = min(abs(targetSNRs - actualSNR));
        
        sisnrIn = computeSISNR(noisy, clean);
        sisnrOut = computeSISNR(denoised, clean);
        sdrIn = computeSDR(noisy, clean);
        sdrOut = computeSDR(denoised, clean);
        
        allResults = [allResults; s, targetSNRs(snrIdx), sisnrOut-sisnrIn, sdrOut-sdrIn];
    end
end

% ==================== 聚合均值 ====================
sisnriShips = nan(numSets, length(targetSNRs));
sdriShips   = nan(numSets, length(targetSNRs));

for s = 1:numSets
    for t = 1:length(targetSNRs)
        mask = (allResults(:,1) == s) & (allResults(:,2) == targetSNRs(t));
        if any(mask)
            sisnriShips(s, t) = mean(allResults(mask, 3));
            sdriShips(s, t)   = mean(allResults(mask, 4));
        end
    end
end

% ==================== 绘图 ====================
grayColors = {[0.00 0.45 0.74], [0.93 0.69 0.13], [0.49 0.18 0.56]};
legendNames = testSets(:, 3);

figure('Units', 'centimeters', 'Position', [2, 2, 32, 16], ...
       'Color', 'white', 'PaperPositionMode', 'auto');

% ---- 左图：SI-SNRi ----
subplot(1, 2, 1);
hold on;
barData = sisnriShips';
b1 = bar(barData, 0.8, 'grouped', 'EdgeColor', 'k', 'LineWidth', 0.6);
for i = 1:length(b1)
    b1(i).FaceColor = grayColors{i};
end
for i = 1:length(b1)
    x = b1(i).XEndPoints;
    y = b1(i).YEndPoints;
    for j = 1:length(y)
        if ~isnan(y(j))
            text(x(j), y(j), sprintf('%.2f', y(j)), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 9, 'Color', 'k');
        end
    end
end
xlabel('输入信噪比 (dB)');
ylabel('SI-SNRi (dB)');
title('(a) SI-SNR 提升量', 'FontWeight', 'bold');
set(gca, 'XTickLabel', targetSNRs);
grid on; box on;
legend(b1, legendNames, 'Location', 'northeast', 'FontSize', 9, 'Box', 'off');

allY1 = sisnriShips(:);
ylim([min(0, min(allY1)*1.1), max(allY1)*1.15]);
hold off;

% ---- 右图：SDRi ----
subplot(1, 2, 2);
hold on;
barData = sdriShips';
b2 = bar(barData, 0.8, 'grouped', 'EdgeColor', 'k', 'LineWidth', 0.6);
for i = 1:length(b2)
    b2(i).FaceColor = grayColors{i};
end
for i = 1:length(b2)
    x = b2(i).XEndPoints;
    y = b2(i).YEndPoints;
    for j = 1:length(y)
        if ~isnan(y(j))
            text(x(j), y(j), sprintf('%.2f', y(j)), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 9, 'Color', 'k');
        end
    end
end
xlabel('输入信噪比 (dB)');
ylabel('SDRi (dB)');
title('(b) SDR 提升量', 'FontWeight', 'bold');
set(gca, 'XTickLabel', targetSNRs);
grid on; box on;
legend(b2, legendNames, 'Location', 'northeast', 'FontSize', 9, 'Box', 'off');

allY2 = sdriShips(:);
ylim([min(allY2)*1.15, max(allY2)*1.15]);
hold off;

% ==================== 保存 ====================
pdfPath = fullfile(figuresDir, 'fig4-5_DCAMF_Net_ShipsEar.pdf');
pngPath = fullfile(figuresDir, 'fig4-5_DCAMF_Net_ShipsEar.png');
exportgraphics(gcf, pdfPath, 'ContentType', 'vector');
saveas(gcf, pngPath);
fprintf('图片已保存至: %s\n', figuresDir);

% ==================== 打印指标汇总 ====================
fprintf('\n========== 指标汇总 ==========\n');
for s = 1:numSets
    for t = 1:length(targetSNRs)
        fprintf('%-25s %-5d %-10.2f %-10.2f\n', testSets{s,1}, targetSNRs(t), sisnriShips(s,t), sdriShips(s,t));
    end
end

% ==================== 辅助函数 ====================
function s = computeSISNR(est, ref)
    est = est(:) - mean(est(:));
    ref = ref(:) - mean(ref(:));
    dotProd = sum(est .* ref);
    sTarget = dotProd * ref / (sum(ref.^2) + 1e-8);
    eNoise = est - sTarget;
    s = 10 * log10(sum(sTarget.^2) / (sum(eNoise.^2) + 1e-8) + 1e-8);
end

function s = computeSDR(est, ref)
    est = est(:); ref = ref(:);
    noise = ref - est;
    s = 10 * log10(sum(ref.^2) / (sum(noise.^2) + 1e-8) + 1e-8);
end