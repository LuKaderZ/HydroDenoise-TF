%% IEEE Paper Figures — DCAMF-Net
%  Generates Fig 2–5 in English, color, IEEE journal style
clear; clc; close all;

% ==================== Paths ====================
projectRoot = 'C:\Users\XUWEILUN\Desktop\HydroDenoise-TF';
passengerDir = fullfile(projectRoot, 'raw_data', 'ShipsEar', 'passenger');
cleanDir = fullfile(projectRoot, 'data', 'ShipsEar', 'test1', 'clean');
noisyDir = fullfile(projectRoot, 'data', 'ShipsEar', 'test1', 'noisy');
crnEstDir = fullfile(projectRoot, 'baselines', 'CRN-causal', 'data', 'data', 'datasets', 'tt', 'tt_test1');
convtasnetEstDir = fullfile(projectRoot, 'experiments', 'conv_tasnet', 'estimates', 'tt_test1');
dprnnEstDir = fullfile(projectRoot, 'experiments', 'dprnn', 'estimates', 'tt_test1');
dcamfEstDir = fullfile(projectRoot, 'experiments', 'dcamf_net', 'denoised', 'ShipsEar_test1');

% DCAMF-Net generalization data
testSets = {
    fullfile(projectRoot, 'data', 'ShipsEar', 'test1'), 'ShipsEar_test1';
    fullfile(projectRoot, 'data', 'ShipsEar', 'test2'), 'ShipsEar_test2';
    fullfile(projectRoot, 'data', 'ShipsEar', 'test3'), 'ShipsEar_test3';
};
denoisedBase = fullfile(projectRoot, 'experiments', 'dcamf_net', 'denoised');

% Fusion weight logs
logDir = fullfile(projectRoot, 'experiments', 'mask_fusion_weights');

% Output
outDir = fullfile(projectRoot, 'figures', 'ieee');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% ==================== Common params ====================
fs = 16000;
nTopPeaks = 5;

% Color palette (colorblind-friendly + print-safe)
cClean   = [0.00 0.00 0.00];   % black
cNoisy   = [0.60 0.60 0.60];   % gray
cCRN     = [0.85 0.33 0.10];   % orange
cCT      = [0.00 0.45 0.74];   % blue
cDP      = [0.47 0.67 0.19];   % green
cDC      = [0.64 0.08 0.18];   % maroon

% ==================== Step 1: detect line spectra (shared) ====================
promFreqs = find_prominent_line_spectra(passengerDir, fs, [0 4000], nTopPeaks);
fprintf('Key line frequencies: %s Hz\n', mat2str(round(promFreqs)));

% ==================== FIGURE 2: Overall PSD Comparison ====================
fprintf('\n--- Fig 2: Overall PSD Comparison ---\n');

bestIdxPSD = select_best_sample_sisnr(cleanDir, noisyDir, dcamfEstDir, fs);
fprintf('Sample index: %d\n', bestIdxPSD);

cleanFiles = dir(fullfile(cleanDir, '*.wav'));
fname = cleanFiles(bestIdxPSD).name;
[clean, ~] = audioread(fullfile(cleanDir, fname)); clean = mean(clean,2);
[noisy, ~] = audioread(fullfile(noisyDir, fname)); noisy = mean(noisy,2);
crn = load_est(crnEstDir, bestIdxPSD, 1);
ct  = load_est(convtasnetEstDir, bestIdxPSD, 2);
dp  = load_est(dprnnEstDir, bestIdxPSD, 2);
dc  = load_est(dcamfEstDir, bestIdxPSD, 3);

minLen = min([length(clean), length(noisy), length(crn), length(ct), length(dp), length(dc)]);
clean=clean(1:minLen); noisy=noisy(1:minLen);
crn=crn(1:minLen); ct=ct(1:minLen); dp=dp(1:minLen); dc=dc(1:minLen);

[pxx_clean, f] = pwelch(clean, hamming(1024), 512, 1024, fs);
pxx_noisy = pwelch(noisy, hamming(1024), 512, 1024, fs);
pxx_crn   = pwelch(crn, hamming(1024), 512, 1024, fs);
pxx_ct    = pwelch(ct,  hamming(1024), 512, 1024, fs);
pxx_dp    = pwelch(dp,  hamming(1024), 512, 1024, fs);
pxx_dc    = pwelch(dc,  hamming(1024), 512, 1024, fs);

idxF = (f >= 0) & (f <= 4000); f_kHz = f(idxF) / 1000;
psd = @(pxx) 10*log10(pxx(idxF));

allPSD = [psd(pxx_clean); psd(pxx_noisy); psd(pxx_crn); psd(pxx_ct); psd(pxx_dp); psd(pxx_dc)];
yMin = floor(min(allPSD)/10)*10; yMax = ceil(max(allPSD)/10)*10;

modelsPSD = {'CRN', psd(pxx_crn); 'Conv-TasNet', psd(pxx_ct); 'DPRNN', psd(pxx_dp); 'DCAMF-Net', psd(pxx_dc)};

figure('Units', 'centimeters', 'Position', [2 2 22 16], 'Color', 'white');
for i = 1:4
    subplot(2, 2, i); hold on;
    plot(f_kHz, psd(pxx_noisy), 'Color', cNoisy, 'LineWidth', 0.6);
    plot(f_kHz, psd(pxx_clean), 'k-', 'LineWidth', 1.5);
    plot(f_kHz, modelsPSD{i,2}, '--', 'Color', cDC, 'LineWidth', 1.2);
    for j = 1:length(promFreqs)
        xline(promFreqs(j)/1000, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.4);
    end
    xlabel('Frequency (kHz)'); ylabel('PSD (dB/Hz)');
    title(modelsPSD{i,1}, 'FontWeight', 'bold');
    xlim([0 4]); ylim([yMin yMax]); grid on; box on;
    legend('Noisy', 'Clean', 'Enhanced', 'Location', 'southwest', 'FontSize', 7);
    hold off;
end
exportgraphics(gcf, fullfile(outDir, 'Fig2_PSD_Comparison.pdf'), 'ContentType', 'vector');
saveas(gcf, fullfile(outDir, 'Fig2_PSD_Comparison.png'));
fprintf('Saved Fig 2\n');

% ==================== FIGURE 3: Line Spectrum Power Recovery ====================
fprintf('\n--- Fig 3: Line Spectrum Recovery ---\n');

bestIdxLine = select_best_sample(cleanDir, noisyDir, crnEstDir, convtasnetEstDir, ...
                                 dprnnEstDir, dcamfEstDir, promFreqs, fs);
fprintf('Sample index: %d\n', bestIdxLine);

fname = cleanFiles(bestIdxLine).name;
[clean, ~] = audioread(fullfile(cleanDir, fname)); clean = mean(clean,2);
[noisy, ~] = audioread(fullfile(noisyDir, fname)); noisy = mean(noisy,2);
crn = load_est(crnEstDir, bestIdxLine, 1);
ct  = load_est(convtasnetEstDir, bestIdxLine, 2);
dp  = load_est(dprnnEstDir, bestIdxLine, 2);
dc  = load_est(dcamfEstDir, bestIdxLine, 3);

minLen = min([length(clean), length(noisy), length(crn), length(ct), length(dp), length(dc)]);
clean=clean(1:minLen); noisy=noisy(1:minLen);
crn=crn(1:minLen); ct=ct(1:minLen); dp=dp(1:minLen); dc=dc(1:minLen);

[pxx_clean, f] = pwelch(clean, hamming(2048), 1024, 4096, fs);
pxx_crn = pwelch(crn, hamming(2048), 1024, 4096, fs);
pxx_ct  = pwelch(ct,  hamming(2048), 1024, 4096, fs);
pxx_dp  = pwelch(dp,  hamming(2048), 1024, 4096, fs);
pxx_dc  = pwelch(dc,  hamming(2048), 1024, 4096, fs);

nF = length(promFreqs);
powers = zeros(nF, 5);
freqLabels = cell(1, nF);
for i = 1:nF
    [~, idxF] = min(abs(f - promFreqs(i)));
    idxWin = max(1, idxF-4):min(length(f), idxF+4);
    powers(i,1) = mean(10*log10(pxx_clean(idxWin)));
    powers(i,2) = mean(10*log10(pxx_crn(idxWin)));
    powers(i,3) = mean(10*log10(pxx_ct(idxWin)));
    powers(i,4) = mean(10*log10(pxx_dp(idxWin)));
    powers(i,5) = mean(10*log10(pxx_dc(idxWin)));
    freqLabels{i} = sprintf('%d Hz', round(promFreqs(i)));
end
dev = powers(:,2:5) - powers(:,1);

figure('Units', 'centimeters', 'Position', [2 2 18 10], 'Color', 'white');
x = 1:nF; width = 0.2;
colors = {cCRN, cCT, cDP, cDC};
modelNames = {'CRN', 'Conv-TasNet', 'DPRNN', 'DCAMF-Net'};
hold on;
bars = cell(1, 4);
for i = 1:4
    bars{i} = bar(x + (i-2.5)*width, dev(:,i), width, ...
        'FaceColor', colors{i}, 'EdgeColor', 'k', 'LineWidth', 0.5);
end
for i = 1:4
    for j = 1:nF
        val = dev(j,i);
        if abs(val) > 0.5
            text(x(j)+(i-2.5)*width, val, sprintf('%.1f', val), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                'FontSize', 6.5, 'Color', colors{i});
        end
    end
end
set(gca, 'XTick', x, 'XTickLabel', freqLabels);
xlabel('Key Line Frequency'); ylabel('Power Deviation from Clean (dB)');
title('Line Spectrum Power Recovery (0 dB = Perfect)', 'FontWeight', 'bold');
legend([bars{:}], modelNames, 'Location', 'best', 'Box', 'off', 'FontSize', 8);
yline(0, 'k-', 'LineWidth', 1.0);
grid on; box on; hold off;
exportgraphics(gcf, fullfile(outDir, 'Fig3_Line_Spectra.pdf'), 'ContentType', 'vector');
saveas(gcf, fullfile(outDir, 'Fig3_Line_Spectra.png'));
fprintf('Saved Fig 3\n');

% ==================== FIGURE 4: Generalization Performance ====================
fprintf('\n--- Fig 4: Generalization Performance ---\n');

targetSNRs = [-15, -10, -5];
numSets = size(testSets, 1);
allResults = [];

for s = 1:numSets
    cleanD = fullfile(testSets{s,1}, 'clean');
    noisyD = fullfile(testSets{s,1}, 'noisy');
    denoisedD = fullfile(denoisedBase, testSets{s,2});
    if ~exist(denoisedD, 'dir'), continue; end
    files = dir(fullfile(cleanD, '*.wav'));
    [~, idx] = sort({files.name}); files = files(idx);
    for ff = 1:length(files)
        fn = files(ff).name;
        dpth = fullfile(denoisedD, fn);
        if ~exist(dpth, 'file'), continue; end
        [cleanSig, ~] = audioread(fullfile(cleanD, fn));
        noisySig = audioread(fullfile(noisyD, fn));
        denoisedSig = audioread(dpth);
        if size(cleanSig,2)>1, cleanSig = mean(cleanSig,2); end
        if size(noisySig,2)>1, noisySig = mean(noisySig,2); end
        if size(denoisedSig,2)>1, denoisedSig = mean(denoisedSig,2); end
        minL = min([length(cleanSig), length(noisySig), length(denoisedSig)]);
        cleanSig=cleanSig(1:minL); noisySig=noisySig(1:minL); denoisedSig=denoisedSig(1:minL);
        noiseActual = noisySig - cleanSig;
        actualSNR = 10*log10(mean(cleanSig.^2)/(mean(noiseActual.^2)+1e-10));
        [~, snrIdx] = min(abs(targetSNRs - actualSNR));
        sisnrIn  = compute_sisnr(noisySig, cleanSig);
        sisnrOut = compute_sisnr(denoisedSig, cleanSig);
        sdrIn  = compute_sdr(noisySig, cleanSig);
        sdrOut = compute_sdr(denoisedSig, cleanSig);
        allResults = [allResults; s, targetSNRs(snrIdx), sisnrOut-sisnrIn, sdrOut-sdrIn];
    end
end

sisnriShips = nan(numSets, length(targetSNRs));
sdriShips   = nan(numSets, length(targetSNRs));
for s = 1:numSets
    for t = 1:length(targetSNRs)
        mask = (allResults(:,1)==s) & (allResults(:,2)==targetSNRs(t));
        if any(mask)
            sisnriShips(s,t) = mean(allResults(mask,3));
            sdriShips(s,t)   = mean(allResults(mask,4));
        end
    end
end

grayColors = {[0.25 0.25 0.25], [0.55 0.55 0.55], [0.85 0.85 0.85]};
legendNames = {'Test-1 (Known + Known)', 'Test-2 (Unseen Vessel)', 'Test-3 (Unseen Noise)'};

figure('Units', 'centimeters', 'Position', [2 2 30 13], 'Color', 'white');

subplot(1,2,1); hold on;
b1 = bar(sisnriShips', 0.8, 'grouped', 'EdgeColor', 'k', 'LineWidth', 0.6);
for i = 1:length(b1), b1(i).FaceColor = grayColors{i}; end
for i = 1:length(b1)
    for j = 1:length(b1(i).YEndPoints)
        if ~isnan(b1(i).YEndPoints(j))
            text(b1(i).XEndPoints(j), b1(i).YEndPoints(j), sprintf('%.2f', b1(i).YEndPoints(j)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8);
        end
    end
end
xlabel('Input SNR (dB)'); ylabel('SI-SNRi (dB)');
title('(a) SI-SNR Improvement', 'FontWeight', 'bold');
set(gca, 'XTickLabel', targetSNRs); grid on; box on;
legend(b1, legendNames, 'Location', 'northeast', 'FontSize', 8, 'Box', 'off');
hold off;

subplot(1,2,2); hold on;
b2 = bar(sdriShips', 0.8, 'grouped', 'EdgeColor', 'k', 'LineWidth', 0.6);
for i = 1:length(b2), b2(i).FaceColor = grayColors{i}; end
for i = 1:length(b2)
    for j = 1:length(b2(i).YEndPoints)
        if ~isnan(b2(i).YEndPoints(j))
            text(b2(i).XEndPoints(j), b2(i).YEndPoints(j), sprintf('%.2f', b2(i).YEndPoints(j)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8);
        end
    end
end
xlabel('Input SNR (dB)'); ylabel('SDRi (dB)');
title('(b) SDR Improvement', 'FontWeight', 'bold');
set(gca, 'XTickLabel', targetSNRs); grid on; box on;
legend(b2, legendNames, 'Location', 'northeast', 'FontSize', 8, 'Box', 'off');
hold off;
exportgraphics(gcf, fullfile(outDir, 'Fig4_Generalization.pdf'), 'ContentType', 'vector');
saveas(gcf, fullfile(outDir, 'Fig4_Generalization.png'));
fprintf('Saved Fig 4\n');

% ==================== FIGURE 5: Fusion Weight Distribution ====================
fprintf('\n--- Fig 5: Fusion Weights ---\n');

logFiles = {
    fullfile(logDir, 'train_avg.log');
    fullfile(logDir, 'train_low.log');
    fullfile(logDir, 'train_high.log');
};
groupNames = {'Average SNR', 'Low SNR', 'High SNR'};
numGroups = length(groupNames);
numLayers = 10;
weights = zeros(numGroups, numLayers);
for g = 1:numGroups
    w = extract_fusion_weights(logFiles{g});
    if length(w) == numLayers, weights(g,:) = w;
    else, weights(g,:) = nan(1, numLayers); end
end

figure('Units', 'centimeters', 'Position', [2 2 18 8], 'Color', 'white');
barData = weights';
b = bar(barData, 0.8, 'grouped', 'EdgeColor', 'k', 'LineWidth', 0.6);
for i = 1:length(b)
    b(i).FaceColor = grayColors{i};
    b(i).BarWidth = 0.8;
end
for i = 1:length(b)
    for j = 1:length(b(i).YEndPoints)
        if b(i).YEndPoints(j) > 0.01
            text(b(i).XEndPoints(j), b(i).YEndPoints(j), sprintf('%.2f', b(i).YEndPoints(j)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 7);
        end
    end
end
xlabel('DCAM Block Layer'); ylabel('Fusion Weight (softmax)');
title('Multi-Layer Mask Fusion Weights under Different SNR Conditions', 'FontWeight', 'bold');
set(gca, 'XTickLabel', 1:numLayers);
grid on; set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.4);
box on; legend(groupNames, 'Location', 'northeast', 'FontSize', 9, 'Box', 'off');
exportgraphics(gcf, fullfile(outDir, 'Fig5_Fusion_Weights.pdf'), 'ContentType', 'vector');
saveas(gcf, fullfile(outDir, 'Fig5_Fusion_Weights.png'));
fprintf('Saved Fig 5\n');

fprintf('\n=== All figures saved to %s ===\n', outDir);

% ==================== Helper Functions ====================

function freqs = find_prominent_line_spectra(audioDir, fs, freqRange, nPeaks)
    files = dir(fullfile(audioDir, '*.wav'));
    if isempty(files), error('No WAV files in %s', audioDir); end
    avgPxx = []; count = 0;
    for k = 1:length(files)
        [sig, sr] = audioread(fullfile(audioDir, files(k).name));
        sig = mean(sig,2);
        if sr ~= fs, sig = resample(sig, fs, sr); end
        [pxx, f] = pwelch(sig, hamming(2048), 1024, 4096, fs);
        if isempty(avgPxx), avgPxx = pxx; else, avgPxx = avgPxx + pxx; end
        count = count + 1;
    end
    avgPxx = avgPxx / count;
    avgPxx_dB = 10*log10(avgPxx);
    idxRange = (f >= freqRange(1)) & (f <= freqRange(2));
    [pks, locs] = findpeaks(avgPxx_dB(idxRange), f(idxRange), ...
        'MinPeakProminence', 5, 'MinPeakDistance', 10);
    if length(locs) > nPeaks
        [~, sortIdx] = sort(pks, 'descend');
        locs = locs(sortIdx(1:nPeaks));
    end
    freqs = sort(locs);
end

function bestIdx = select_best_sample_sisnr(cleanDir, noisyDir, dcamfDir, fs)
    cleanFiles = dir(fullfile(cleanDir, '*.wav'));
    nSamples = length(cleanFiles);
    bestSIi = -inf; bestIdx = 1; analysisWin = 0.05;
    for k = 1:nSamples
        fname = cleanFiles(k).name;
        dcamfFile = fullfile(dcamfDir, sprintf('%06d.wav', k));
        if ~exist(dcamfFile, 'file'), continue; end
        [clean, ~] = audioread(fullfile(cleanDir, fname)); clean = mean(clean,2);
        [noisy, ~] = audioread(fullfile(noisyDir, fname)); noisy = mean(noisy,2);
        dcamf = mean(audioread(dcamfFile), 2);
        minLen = min([length(clean), length(noisy), length(dcamf)]);
        clean=clean(1:minLen); noisy=noisy(1:minLen); dcamf=dcamf(1:minLen);
        winLen = round(analysisWin*fs);
        if minLen < winLen, continue; end
        nWindows = minLen - winLen + 1;
        energies = zeros(nWindows, 1);
        for s = 1:winLen:nWindows
            energies(s) = sum(noisy(s:s+winLen-1).^2);
        end
        [~, maxIdx] = max(energies);
        startS = max(1, maxIdx);
        endS = min(minLen, startS+winLen-1);
        if endS-startS+1 < winLen, continue; end
        sisnr_in = compute_sisnr(noisy(startS:endS), clean(startS:endS));
        sisnr_out = compute_sisnr(dcamf(startS:endS), clean(startS:endS));
        si_snri = sisnr_out - sisnr_in;
        if si_snri > bestSIi, bestSIi = si_snri; bestIdx = k; end
    end
    if bestIdx == 0, bestIdx = 1; end
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
            err_crn = abs(mean(10*log10(pxx_crn(idxWin)))-ref);
            err_ct  = abs(mean(10*log10(pxx_ct(idxWin)))-ref);
            err_dp  = abs(mean(10*log10(pxx_dp(idxWin)))-ref);
            err_dc  = abs(mean(10*log10(pxx_dc(idxWin)))-ref);
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
    else, est = []; return;
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

function w = extract_fusion_weights(logFilePath)
    if ~exist(logFilePath, 'file')
        error('Log file not found: %s', logFilePath);
    end
    fid = fopen(logFilePath, 'r');
    lines = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
    fclose(fid);
    lines = lines{1};
    idx = [];
    for i = length(lines):-1:1
        if contains(lines{i}, 'MaskFusion Weights (softmax):')
            idx = i; break;
        end
    end
    if isempty(idx), error('Fusion weight line not found in %s.', logFilePath); end
    targetLine = lines{idx};
    tokens = regexp(targetLine, 'MaskFusion Weights \(softmax\):\s*(\[.*\])', 'tokens');
    if isempty(tokens), error('Cannot parse weight line: %s', targetLine); end
    w = str2num(tokens{1}{1});
    if isempty(w), error('Weight array parse failed: %s', tokens{1}{1}); end
end
