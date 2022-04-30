function gslt = getForwardKinematicsMap(xi_1, xi_2, xi_3, theta_1, theta_2, theta_3, gsl0, linkEnd)
    if linkEnd == 1
        gslt = expm(getTwistMatrix(xi_1).*theta_1) * gsl0 ;
    end
    if linkEnd == 2
        gslt = expm(getTwistMatrix(xi_1).*theta_1) * expm(getTwistMatrix(xi_2).*theta_2) * gsl0 ;
    end
    if linkEnd == 3
        gslt = expm(getTwistMatrix(xi_1).*theta_1) * expm(getTwistMatrix(xi_2).*theta_2) * expm(getTwistMatrix(xi_3).*theta_3) * gsl0 ;
    end
end