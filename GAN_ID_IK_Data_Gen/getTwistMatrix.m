function xi_hat = getTwistMatrix(xi)
    xi_hat = [0, -xi(6), xi(5), xi(1); xi(6), 0, -xi(4), xi(2); -xi(5), xi(4), 0, xi(3); 0 0 0 0];
end