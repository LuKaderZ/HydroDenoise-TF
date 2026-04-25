%% DCAMF-Net 指标计算 + 绘图（直接读取降噪音频）
clear; clc; close all;

% ==================== 路径配置 ====================
projectRoot = 'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF';
dataBase = fullfile(projectRoot, 'data');
denoisedBase = fullfile(projectRoot, 'experiments', 'dcamf_net', 'denoised');
figuresDir = fullfile(projectRoot, 'figures');
if ~exist(figuresDir, 'dir'), mkdir(figuresDir); end

% ==================== 参数 ====================
targetSNRs = [-15, -10, -5];

% 测试集定义：{数据子路径, 降噪音频子目录名, 显示名称}
testSets = {
    'ShipsEar/test1', 'ShipsEar_test1', 'Test1 (Known ship + Known noise)';
    'ShipsEar/test2', 'ShipsEar_test2', 'Test2 (Unseen ship + Known noise)';
    'ShipsEar/test3', 'ShipsEar_test3', 'Test3 (Known ship + Unseen noise)';
    'DeepShip/test',  'DeepShip_test',  'DeepShip';
};

% ==================== 计算指标 ====================
allResults = [];  % 存储 [setIdx, snr, sisnri, sdri]

for s = 1:size(testSets, 1)
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
        
        [clean, fs] = audioread(cleanPath);
        noisy = audioread(noisyPath);
        denoised = audioread(denoisedPath);
        
        if size(clean,2)>1, clean = mean(clean,2); end
        if size(noisy,2)>1, noisy = mean(noisy,2); end
        if size(denoised,2)>1, denoised = mean(denoised,2); end
        
        minLen = min([length(clean), length(noisy), length(denoised)]);
        clean = clean(1:minLen); noisy = noisy(1:minLen); denoised = denoised(1:minLen);
        
        % 实际 SNR
        noiseActual = noisy - clean;
        actualSNR = 10 * log10(mean(clean.^2) / (mean(noiseActual.^2) + 1e-10));
        [~, snrIdx] = min(abs(targetSNRs - actualSNR));
        
        % SI-SNR
        sisnrIn = computeSISNR(noisy, clean);
        sisnrOut = computeSISNR(denoised, clean);
        sdrIn = computeSDR(noisy, clean);
        sdrOut = computeSDR(denoised, clean);
        
        allResults = [allResults; s, targetSNRs(snrIdx), sisnrOut-sisnrIn, sdrOut-sdrIn];
    end
end

% ==================== 汇总均值 ====================
numSets = size(testSets, 1) - 1;  % 前3个是 ShipsEar
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

% DeepShip
deepMask = (allResults(:,1) == 4);
if any(deepMask)
    sisnriDeep = mean(allResults(deepMask, 3));
    sdriDeep   = mean(allResults(deepMask, 4));
    hasDeep = true;
else
    hasDeep = false;
end

% ==================== 绘图 ====================
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 11);

figure('Units', 'centimeters', 'Position', [2, 2, 16, 8], ...
       'Color', 'white', 'PaperPositionMode', 'auto');

grayColors = {[0.25 0.25 0.25], [0.55 0.55 0.55], [0.85 0.85 0.85]};
legendNames = testSets(1:3, 3);

% ---- SI-SNRi ----
subplot(1,2,1);
barData = sisnriShips';
b = bar(barData, 0.8, 'grouped', 'EdgeColor', 'k', 'LineWidth', 0.6);
for i = 1:length(b)
    b(i).FaceColor = grayColors{i};
end
xlabel('Input SNR (dB)'); ylabel('SI-SNRi (dB)');
title('(a) SI-SNR Improvement', 'FontWeight', 'bold');
set(gca, 'XTickLabel', targetSNRs);
grid on; box on;
for i = 1:length(b)
    x = b(i).XEndPoints; y = b(i).YEndPoints;
    for j = 1:length(y)
        if ~isnan(y(j))
            text(x(j), y(j), sprintf('%.2f', y(j)), ...
                'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',9);
        end
    end
end

% ---- SDRi ----
subplot(1,2,2);
barData = sdriShips';
b = bar(barData, 0.8, 'grouped', 'EdgeColor', 'k', 'LineWidth', 0.6);
for i = 1:length(b)
    b(i).FaceColor = grayColors{i};
end
xlabel('Input SNR (dB)'); ylabel('SDRi (dB)');
title('(b) SDR Improvement', 'FontWeight', 'bold');
set(gca, 'XTickLabel', targetSNRs);
grid on; box on;
for i = 1:length(b)
    x = b(i).XEndPoints; y = b(i).YEndPoints;
    for j = 1:length(y)
        if ~isnan(y(j))
            text(x(j), y(j), sprintf('%.2f', y(j)), ...
                'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',9);
        end
    end
end

lgd = legend(legendNames, 'Orientation', 'horizontal', 'FontSize', 10, 'Box', 'off');
set(lgd, 'Position', [0.18, 0.01, 0.64, 0.05], 'Units', 'normalized');

% 保存
exportgraphics(gcf, fullfile(figuresDir, 'fig4-3_DCAMF_Net_ShipsEar.pdf'), 'ContentType', 'vector');
saveas(gcf, fullfile(figuresDir, 'fig4-3_DCAMF_Net_ShipsEar.png'));
fprintf('图片已保存至 figures/\n');

% ==================== 打印汇总 ====================
fprintf('\n========== 指标汇总 ==========\n');
for s = 1:numSets
    for t = 1:length(targetSNRs)
        fprintf('%-25s %-5d %-10.2f %-10.2f\n', testSets{s,1}, targetSNRs(t), sisnriShips(s,t), sdriShips(s,t));
    end
end
if hasDeep
    fprintf('%-25s %-5s %-10.2f %-10.2f\n', 'DeepShip', '--', sisnriDeep, sdriDeep);
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