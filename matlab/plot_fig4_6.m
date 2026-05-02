%% 图4-6：不同信噪比下多层掩码融合权重的分布对比
clear; clc; close all;

% ==================== 字体与样式设置 (中文黑体) ====================
set(0, 'DefaultAxesFontName', 'SimHei');
set(0, 'DefaultTextFontName', 'SimHei');
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultLineLineWidth', 1.0);

% ==================== 路径配置 ====================
% 脚本所在目录: matlab/，向上一级到 HydroDenoise-TF/
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fullfile(scriptDir, '..');   % 根据您的实际存放位置调整

logDir = fullfile(projectRoot, 'experiments', 'mask_fusion_weights');
logFiles = {
    fullfile(logDir, 'train_avg.log'),    % 平均SNR组
    fullfile(logDir, 'train_low.log'),    % 低SNR组
    fullfile(logDir, 'train_high.log')    % 高SNR组
};

groupNames = {'平均SNR', '低SNR', '高SNR'};
numGroups = length(groupNames);
numLayers = 10;   % 10层DCAM模块

% ==================== 提取权重数据 ====================
weights = zeros(numGroups, numLayers);
for g = 1:numGroups
    w = extract_fusion_weights(logFiles{g});
    if length(w) == numLayers
        weights(g, :) = w;
    else
        warning('文件 %s 中提取的权重长度不是 %d，请检查。', logFiles{g}, numLayers);
        weights(g, :) = nan(1, numLayers);
    end
end

% ==================== 绘图 ====================
grayColors = {[0.00 0.45 0.74], [0.47 0.67 0.19], [0.64 0.08 0.18]};

figure('Units', 'centimeters', 'Position', [2, 2, 16, 12], ...
       'Color', 'white', 'PaperPositionMode', 'auto');

% 水平分组柱状图
barData = weights';   % 10×3矩阵
b = barh(barData, 0.8, 'grouped', 'EdgeColor', 'k', 'LineWidth', 0.6);
for i = 1:length(b)
    b(i).FaceColor = grayColors{i};
    b(i).BarWidth = 0.8;
end

xlabel('融合权重 (softmax)');
ylabel('DCAM模块层数');
title('不同信噪比下多层掩码融合权重分布', 'FontWeight', 'bold');
set(gca, 'YTickLabel', 1:numLayers);
grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.4, 'GridColor', [0.2 0.2 0.2]);
box on;

% 柱端数值标签（仅当权重大于0.01时显示）
for i = 1:length(b)
    v = b(i).XEndPoints;   % bar values (weights)
    y = b(i).YEndPoints;   % bar positions (layers)
    for j = 1:length(v)
        if v(j) > 0.01
            text(v(j) + 0.01, y(j), sprintf('%.2f', v(j)), ...
                'HorizontalAlignment', 'left', ...
                'VerticalAlignment', 'middle', ...
                'FontSize', 8, 'Color', 'k');
        end
    end
end

legend(groupNames, 'Location', 'northeast', 'FontSize', 10, 'Box', 'off');

% ==================== 保存 ====================
saveDir = fullfile(projectRoot, 'figures');
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
pdfPath = fullfile(saveDir, 'fig4-6_fusion_weights_comparison.pdf');
pngPath = fullfile(saveDir, 'fig4-6_fusion_weights_comparison.png');
exportgraphics(gcf, pdfPath, 'ContentType', 'vector');
saveas(gcf, pngPath);
fprintf('图4-6已保存至 %s\n', saveDir);

% ==================== 辅助函数 ====================
function w = extract_fusion_weights(logFilePath)
    % 从训练日志中提取最后一行的融合权重 (softmax 后)
    if ~exist(logFilePath, 'file')
        error('日志文件不存在: %s', logFilePath);
    end
    % 读取文件所有行
    fid = fopen(logFilePath, 'r');
    lines = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
    fclose(fid);
    lines = lines{1};
    
    % 从后往前找第一个包含 "MaskFusion Weights (softmax):" 的行
    idx = [];
    for i = length(lines):-1:1
        if contains(lines{i}, 'MaskFusion Weights (softmax):')
            idx = i;
            break;
        end
    end
    if isempty(idx)
        error('在文件 %s 中未找到融合权重行。', logFilePath);
    end
    
    targetLine = lines{idx};
    % 提取等号后面的部分，格式如 [0.1, 0.2, ...]
    expr = 'MaskFusion Weights \(softmax\):\s*(\[.*\])';
    tokens = regexp(targetLine, expr, 'tokens');
    if isempty(tokens)
        error('无法解析权重行: %s', targetLine);
    end
    weightStr = tokens{1}{1};
    % 将字符串转换为数值数组
    w = str2num(weightStr); %#ok<ST2NM>
    if isempty(w)
        error('权重数组解析失败: %s', weightStr);
    end
end