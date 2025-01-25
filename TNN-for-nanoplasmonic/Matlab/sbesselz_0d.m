function [ j, h ] = sbesselz_0d( n, x )

    Yj = [];
    Yh = [];
    
    for m = 1 : n
        
        Yj(m, :) = sqrt(pi ./ 2 ./ x) .* besselj(m + 1 / 2, x);
        Yh(m, :) = sqrt(pi ./ 2 ./ x) .* besselh(m + 1 / 2, x);
        
    end
    
    j = Yj;
    h = Yh;
    
end
%%
%     j0 = sin(x) ./ x;
%     j1 = (sin(x) - x .* cos(x)) ./ (x .^ 2);
%     
%     h0 = - 1i .* exp(1i .* x) ./ x;
%     h1 = (- 1i ./ (x .^ 2) - 1 ./ x) .* exp(1i .* x);
%     
%     J = [j0; j1]; H = [h0; h1];
%     
%     for m = 2 : n
% 
%         J(end + 1, :) = (2 .* m + 1) .* J(end, :) ./ x - J(end - 1, :);
%         H(end + 1, :) = (2 .* m + 1) .* H(end, :) ./ x - H(end - 1, :);
%         
%     end
%     
%     j = J(2 : end, :);
%     h = H(2 : end, :);
%     
% end