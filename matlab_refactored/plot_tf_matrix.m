function plot_tf_matrix(time, frequency, matrix, plot_title)
%PLOT_TF_MATRIX 绘制时频矩阵。

figure;
imagesc(time, frequency, matrix);
axis xy;
xlabel('Time/s');
ylabel('Frequency/Hz');
title(plot_title);
colorbar;
end

