test1
   样本数: 1722
   SISNR:  2.1253 dB (SISNRi: 12.1285)
   SDR:    3.5035 dB (SDRi: 12.8363)

test2
   样本数: 1491
   SISNR:  -0.3826 dB (SISNRi: 9.6347)
   SDR:    1.3033 dB (SDRi: 10.6463)

test3
   样本数: 1722
   SISNR:  -10.4399 dB (SISNRi: -0.7370)
   SDR:    -9.6865 dB (SDRi: -0.5845)

DeepShip/test
   样本数: 800
   SISNR:  -7.9781 dB (SISNRi: 2.7568)
   SDR:    10.3294 dB (SDRi: 5.5650)

========== 处理 ShipsEar test1 ==========

========== CRN test1 指标 ==========
  SNR ~-15 dB: SI-SNRi = 6.63 dB, SDRi = 13.34 dB (样本数: 574)
  SNR ~-10 dB: SI-SNRi = 6.64 dB, SDRi = 9.22 dB (样本数: 574)
  SNR ~-5 dB: SI-SNRi = 5.53 dB, SDRi = 4.53 dB (样本数: 574)

填入表4-1 (ShipsEar)：
  SI-SNRi = 6.27 dB
  SDRi    = 9.03 dB

========== 处理 DeepShip ==========

DeepShip 整体指标：
  SI-SNRi = 2.09 dB
  SDRi    = -10.34 dB

填入表4-1 (DeepShip)：
  SI-SNRi = 2.09 dB
  SDRi    = -10.34 dB

========== Conv-TasNet test1:===========
  -15 dB: SI-SNRi=17.38, SDRi=-2.66 (n=574)
  -10 dB: SI-SNRi=13.91, SDRi=-7.46 (n=574)
  -5 dB: SI-SNRi=9.64, SDRi=-12.55 (n=574)
表4-1: SI-SNRi=13.65, SDRi=-7.56

========== DPRNN test1: ================
  -15 dB: SI-SNRi=8.93, SDRi=15.35 (n=574)
  -10 dB: SI-SNRi=7.84, SDRi=11.54 (n=574)
  -5 dB: SI-SNRi=6.23, SDRi=8.26 (n=574)
表4-1: SI-SNRi=7.67, SDRi=11.72

========== 消融实验 ====================
--- 移除全局支路 ---
  -15 dB: SI-SNRi=-0.01, SDRi=-0.02 (n=574)
  -10 dB: SI-SNRi=-0.01, SDRi=-0.02 (n=574)
  -5 dB: SI-SNRi=-0.02, SDRi=-0.03 (n=574)
表4-2: SI-SNR=-0.01, SDR=-0.02

--- 移除局部支路 ---
  -15 dB: SI-SNRi=1.65, SDRi=-17.05 (n=574)
  -10 dB: SI-SNRi=1.44, SDRi=-18.26 (n=574)
  -5 dB: SI-SNRi=1.12, SDRi=-19.60 (n=574)
表4-2: SI-SNR=1.40, SDR=-18.30

--- 移除卷积增强 ---
  -15 dB: SI-SNRi=11.94, SDRi=15.84 (n=574)
  -10 dB: SI-SNRi=10.53, SDRi=13.17 (n=574)
  -5 dB: SI-SNRi=7.88, SDRi=9.71 (n=574)
表4-2: SI-SNR=10.12, SDR=12.91