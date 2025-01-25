function [ Gfs, Fn ] = sphereGreenFunction(R1, Theta1, Phi1, R2, Theta2, Phi2, kappa, mu, Rf, Lm, Ls, Lf)
%% 变量定义
       % [Gfs1, Fn1] = sphereGreenFunction(R,    0,     0,    R,    0,    0,   kappa, 1 , Rf, 3 ,  1,  1);
    rho1 = kappa(Lf, :) .* R1;
    rho2 = kappa(Ls, :) .* R2;
    
    cos_Theta = cos(Theta1) .* cos(Theta2) + sin(Theta1) .* sin(Theta2) .* cos(Phi1 - Phi2);

    Gc = 1i .* kappa(Ls, :) ./ (4 .* pi);

%% 计算格林函数
    
    n_Sum = 23;
    n_All = (1 : n_Sum)';
    
    [ jn1, hn1 ]  = sbesselz_0d( n_Sum, rho1 );
    r_jn1 = n_All .* (n_All + 1) .* jn1 ./ rho1;
    r_hn1 = n_All .* (n_All + 1) .* hn1 ./ rho1;
    
    [ jn2, hn2 ]  = sbesselz_0d( n_Sum, rho2 );
    r_jn2 = n_All .* (n_All + 1) .* jn2 ./ rho2;
    r_hn2 = n_All .* (n_All + 1) .* hn2 ./ rho2;
    
    [Afs_N, Bfs_N, Cfs_N, Dfs_N] = sphereGreenCoefficient(2, n_Sum, kappa, Rf, mu, Lm, Ls, Lf);

    Dum1 = slegendre0(n_Sum, cos_Theta);
    
    Cn = (2 .* n_All + 1) ./ (n_All .* (n_All + 1));
    
    N10 = r_hn1.* r_jn2;
    N11 = r_hn1 .* r_hn2;
    N00 = r_jn1 .* r_jn2;
    N01 = r_jn1 .* r_hn2;
    
    MN3 = (1-K_delta(Lm, Lf)) .* (N10 .* (1-K_delta(1, Ls)) .* Afs_N + N11 .* (1-K_delta(Lm, Ls)) .* Bfs_N);
    MN4 = (1-K_delta(1 , Lf)) .* (N00 .* (1-K_delta(1, Ls)) .* Cfs_N + N01 .* (1-K_delta(Lm, Ls)) .* Dfs_N);
    
    Gs =  sum(Cn .* (MN3 + MN4) .* Dum1);

    G0_1 = sum(Cn .* N10);
    G0_2 = sum(Cn .* N01);

%% 输出结果
    
   G0s = Gc .* (Hv(R1, R2) .* G0_1 + (1 - Hv(R1, R2)) .* G0_2);
   Gfs = Gc .* Gs;
   Fn = imag(Gfs) ./ imag(G0s);
   
end