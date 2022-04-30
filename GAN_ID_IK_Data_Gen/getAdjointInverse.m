function  xiP = getAdjointInverse(A, b)
    xiP = inv(A) * getTwistMatrix(b) * (A);
end