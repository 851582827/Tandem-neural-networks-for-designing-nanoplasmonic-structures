function [ p ] = slegendre0(n, theta)
    
    Pm = [];
    for m = 1 : n
        slgd = legendre(m, theta);
        Pm(m, :) = slgd(1, :);
    end
    p = Pm;
    
end
