function [ j, h, dj, dh ] = sbesselz_1d( n, x )

    n_d = ( 2 : n + 1 )';
    Yj = [];
    Yh = [];
    for m = 0 : n
        
        Yj(m + 1, :) = sqrt(pi ./ 2 ./ x) .* besselj(m + 1 / 2, x);
        Yh(m + 1, :) = sqrt(pi ./ 2 ./ x) .* besselh(m + 1 / 2, x);
        
    end
    j = Yj(2 : end, :);
    h = Yh(2 : end, :);
    dj = Yj(1 : end - 1, :) -  n_d .* Yj(2 : end, :) ./ x;
    dh = Yh(1 : end - 1, :) -  n_d .* Yh(2 : end, :) ./ x;
    
end
%%
%     Yj = [];
%     Yh = [];
%     
%     for m = 0 : (n + 1)
%         Yj(m + 1, :) = sqrt(pi ./ 2 ./ x) .* besselj(m + 1 / 2, x);
%         Yh(m + 1, :) = sqrt(pi ./ 2 ./ x) .* besselh(m + 1 / 2, x);
%     end
%     
%     j = Yj(2 : end - 1, :);
%     h = Yh(2 : end - 1, :);
%     dj = (Yj(1 : end - 2, :) - Yj(3 : end, :)) ./ 2;
%     dh = (Yh(1 : end - 2, :) - Yh(3 : end, :)) ./ 2;
%     
% end
%%
%     j0 = sin(x) ./ x;
%     j1 = (sin(x) - x .* cos(x)) ./ (x .^ 2);
%     
%     h0 = - 1i .* exp(1i .* x) ./ x;
%     h1 = (- 1i ./ (x .^ 2) - 1 ./ x) .* exp(1i .* x);
%     
%     J = [j0; j1]; H = [h0; h1];
%     
%     for m = 2 : n + 1
%         
%         J(end+1, :) = (2 .* m + 1) ./ x .* J(end, :) - J(end-1, :);
%         H(end+1, :) = (2 .* m + 1) ./ x .* H(end, :) - H(end-1, :);
%         
%     end
%     
%     j = J(2 : end - 1, :);
%     h = H(2 : end - 1, :);
%     dj = (J(1 : end - 2, :) - J(3 : end, :)) ./ 2;
%     dh = (H(1 : end - 2, :) - H(3 : end, :)) ./ 2;
%     
% end

