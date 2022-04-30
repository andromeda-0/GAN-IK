fig = figure('visible', 'off');
grid on
hold on

for max_steps = [1e2, 1e3, 1e4, 1e5, 1e6]
    max_steps = int32(max_steps); %#ok<FXSET>
    mse = IK_func(max_steps, 10);


    RMSE = sqrt(mean(mse));
    plot(sqrt(mse))
    fprintf('Max Steps: %d, Pseudoinverse: RMSE is %.3f\n', max_steps, RMSE);
    disp(mse);
end

legend('1e2', '1e3', '1e4', '1e5', '1e6')
saveas(fig, 'IK-IP.png')