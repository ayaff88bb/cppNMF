function plot_input_analysis(data)
%PLOT_INPUT_ANALYSIS 绘制输入信号、频谱和包络谱。

figure;
plot(data.time, data.signal);
xlabel('Time/s');
ylabel('Amplitude');
title('Time-domain Signal');

figure;
subplot(2, 1, 1);
plot(data.frequency_half, data.spectrum);
xlabel('Frequency/Hz');
ylabel('Amplitude');
title('Spectrum');

subplot(2, 1, 2);
plot(data.frequency_half, data.envelope_spectrum);
xlabel('Frequency/Hz');
ylabel('Amplitude');
title('Envelope Spectrum');
linkaxes(findobj(gcf, 'Type', 'axes'), 'x');
end

