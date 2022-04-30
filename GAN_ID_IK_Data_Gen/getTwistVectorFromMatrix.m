function xi = getTwistVectorFromMatrix(xi_hat)
    xi = [xi_hat(1,4); xi_hat(2,4);xi_hat(3,4) ;xi_hat(3,2); xi_hat(1,3); xi_hat(2,1)];
end