function [ ft, xt ] = IFFT( Fw, dw, xw, dt, N)
    
    ft1= []; 
    for n_t = (0 : N - 1)
        E_wt = exp(1i .* n_t .* dt .* xw );
        ft1(1, end + 1) = sum(Fw .* E_wt .* dw) ./ (2 .* pi);
    end
    
    ft = ft1;
    xt = (0 : N - 1) .* dt;

end

