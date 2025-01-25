function [ epsilon_Ag ] = epseilonAg( omega_eV, d )
%-epsilonAg是银金属的相对电容率
%-omega是响应角频率
%-d是厚度

    e2r = importdata('epsr.txt');
    e2rx = e2r(:, 1)'; e2ry = e2r(:, 2)';
    E2r = interp1(e2rx, e2ry, omega_eV, 'spline'); % 实部
    
    e2i = importdata('epsi.txt');
    e2ix = e2i(:, 1)'; e2iy = e2i(:, 2)';
    E2i = interp1(e2ix, e2iy, omega_eV, 'spline'); % 虚部
    
    h = 6.62607015e-34; % 普朗克常数
    hba = h ./ (2 .* pi); % 约化普朗克常数
    eq = 1.60217733e-19;
    
%     A = 0.25; 
    A = 0.0;
    
    vf = 1.39e6 .* hba ./ eq;
    Gamma_b = 0.03e15 .* hba ./ eq; % bulk
    omega_p = 1.3987e16 .* hba ./ eq; % ωp
%     Gamma = Gamma_b + A .* vf ./ (d .* 5 .* 1e-9);
    Gamma = Gamma_b + A .* vf ./ (d .* 1e-9);
    % 等离子体频率+表面粗糙度因子的影响
    epsilon_Ag = E2r + 1i .* E2i + omega_p .^ 2 ./ ( omega_eV .^ 2 + 1i .* omega_eV .* Gamma_b) - omega_p .^ 2 ./ (omega_eV .^ 2 + 1i .* omega_eV .* Gamma);
    
end

