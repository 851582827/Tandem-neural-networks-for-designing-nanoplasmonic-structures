function [ A, B, C, D ] = sphereGreenCoefficient(MN, n, kappa, r, mu, Lm, Ls, Lf)
%[Afs_N, Bfs_N, Cfs_N, Dfs_N] = sphereGreenCoefficient(2, n_Sum, kappa, Rf, mu, Lm, Ls, Lf);
%% 定义 mu_f, mu_s, kappa_f, kappa_s

    mu_f = mu(1 : end-1, :);
    kappa_f = kappa(1 : end-1, :);

    mu_s = mu(2 : end, :);
    kappa_s = kappa(2 : end, :);
 
%% 定义 rho_f, rho_s

    rho_f = kappa_f .* r;
    rho_s = kappa_s .* r;
    
%% 初始化表达式

    Am1 = 1; Bm1 = 1; Cm1 = 0; Dm1 = 0;
    Am2 = 0; Bm2 = 0; Cm2 = 0; Dm2 = 0;
    
    Af1 = 1; Bf1 = 1; Cf1 = 0; Df1 = 0;
    Af2 = 0; Bf2 = 0; Cf2 = 0; Df2 = 0;
    
%% 循环递推表达式

    for L = 2 : Lm
        
        [ Jff, Hff, dJff, dHff ] = sbesselz_1d( n, rho_f(L-1, :) );
        dJff = Jff ./ rho_f(L-1, :)  + dJff;
        dHff = Hff ./ rho_f(L-1, :)  + dHff;
        [ Jsf, Hsf, dJsf, dHsf ] = sbesselz_1d( n, rho_s(L-1, :) );
        dJsf = Jsf ./ rho_s(L-1, :)  + dJsf;
        dHsf = Hsf ./ rho_s(L-1, :)  + dHsf;
  
        kp_f = kappa_f(L-1, :);
        kp_s = kappa_s(L-1, :);
        m_f = mu_f(L-1, :);
        m_s = mu_s(L-1, :);
        
        
        if MN == 1
            
            Tf = (m_f .* kp_s .* (Hsf .* dJsf - Jsf .* dHsf)) ./ (m_f .* kp_s .* Hff .* dJsf - m_s .* kp_f .* Jsf .* dHff);
            Tp = (m_f .* kp_s .* (Hsf .* dJsf - Jsf .* dHsf)) ./ (m_f .* kp_s .* Jff .* dHsf - m_s .* kp_f .* Hsf .* dJff);
            
            Rf = (m_f .* kp_s .* Jff .* dJsf - m_s .* kp_f .* Jsf .* dJff) ./ (m_f .* kp_s .* Hff .* dJsf - m_s .* kp_f .* Jsf .* dHff);
            Rp = (m_f .* kp_s .* Hff .* dHsf - m_s .* kp_f .* Hsf .* dHff) ./ (m_f .* kp_s .* Jff .* dHsf - m_s .* kp_f .* Hsf .* dJff);
            
        elseif MN == 2
            
            Tf = (m_f .* kp_s .* (Jsf .* dHsf - Hsf .* dJsf)) ./ (m_f .* kp_s .* Jsf .* dHff - m_s .* kp_f .* Hff .* dJsf);
            Tp = (m_f .* kp_s .* (Hsf .* dJsf - Jsf .* dHsf)) ./ (m_f .* kp_s .* Hsf .* dJff - m_s .* kp_f .* Jff .* dHsf);
            
            Rf = (m_f .* kp_s .* Jsf .* dJff - m_s .* kp_f .* Jff .* dJsf) ./ (m_f .* kp_s .* Jsf .* dHff - m_s .* kp_f .* Hff .* dJsf);
            Rp = (m_f .* kp_s .* Hsf .* dHff - m_s .* kp_f .* Hff .* dHsf) ./ (m_f .* kp_s .* Hsf .* dJff - m_s .* kp_f .* Jff .* dHsf);
            
        end
        
        T1 = 1 ./ Tf; T2 = Rf ./ Tf; T3 = Rp ./ Tp; T4 = 1 ./ Tp;
        
        Al1 = Am1; Bl1 = Bm1; Cl1 = Cm1; Dl1 = Dm1;
        Al2 = Am2; Bl2 = Bm2; Cl2 = Cm2; Dl2 = Dm2;
            
        Am1 = T1 .* Al1 + T2 .* Cl1;
        Am2 = T1 .* Al2 + T2 .* Cl2 - K_delta(L, Ls);
        Cm1 = T3 .* Al1 + T4 .* Cl1;
        Cm2 = T3 .* Al2 + T4 .* Cl2;
            
        Bm1 = T1 .* Bl1 + T2 .* Dl1;
        Bm2 = T1 .* Bl2 + T2 .* Dl2 + T2 .* K_delta((L - 1), Ls);
        Dm1 = T3 .* Bl1 + T4 .* Dl1;
        Dm2 = T3 .* Bl2 + T4 .* Dl2 + T4 .* K_delta((L - 1), Ls);
            
            
        if L == Lf
            Af1 = Am1; Bf1 = Bm1; Cf1 = Cm1; Df1 = Dm1;
            Af2 = Am2; Bf2 = Bm2; Cf2 = Cm2; Df2 = Dm2;
        end
        
    end

    A = Af1 .* (-Am2 ./ Am1) + Af2;
    B = Bf1 .* (-Bm2 ./ Bm1) + Bf2;
    C = Cf1 .* (-Am2 ./ Am1) + Cf2;
    D = Df1 .* (-Bm2 ./ Bm1) + Df2;
    
    
end 