function [Y] = K_delta(m, n)

    y = (m == n);
    Y = double(y);
    
end