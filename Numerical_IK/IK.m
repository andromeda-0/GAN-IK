fig = figure('visible', 'off');
grid on
hold on

for max_steps = [1e2, 1e3, 1e4, 1e5, 1e6]
    max_steps = int32(max_steps); %#ok<FXSET> 
    [mse, percentage_converged] = IK_func(max_steps, 1000);


    RMSE = sqrt(mean(mse));
    scatter(sqrt(mse))
    fprintf('Max Steps: %d, Pseudoinverse: RMSE is %.3f, %.2f%% converged\n', max_steps, RMSE, percentage_converged * 100);
end

legend('1e2', '1e3', '1e4', '1e5', '1e6')
saveas(fig, 'IK-IP.png')