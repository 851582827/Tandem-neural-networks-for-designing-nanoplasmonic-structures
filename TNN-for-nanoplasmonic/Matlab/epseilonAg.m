function [ epsilon_Ag ] = epseilonAg( omega_eV, d )
%-epsilonAg������������Ե�����
%-omega����Ӧ��Ƶ��
%-d�Ǻ��

    e2r = importdata('epsr.txt');
    e2rx = e2r(:, 1)'; e2ry = e2r(:, 2)';
    E2r = interp1(e2rx, e2ry, omega_eV, 'spline'); % ʵ��
    
    e2i = importdata('epsi.txt');
    e2ix = e2i(:, 1)'; e2iy = e2i(:, 2)';
    E2i = interp1(e2ix, e2iy, omega_eV, 'spline'); % �鲿
    
    h = 6.62607015e-34; % ���ʿ˳���
    hba = h ./ (2 .* pi); % Լ�����ʿ˳���
    eq = 1.60217733e-19;
    
%     A = 0.25; 
    A = 0.0;
    
    vf = 1.39e6 .* hba ./ eq;
    Gamma_b = 0.03e15 .* hba ./ eq; % bulk
    omega_p = 1.3987e16 .* hba ./ eq; % ��p
%     Gamma = Gamma_b + A .* vf ./ (d .* 5 .* 1e-9);
    Gamma = Gamma_b + A .* vf ./ (d .* 1e-9);
    % ��������Ƶ��+����ֲڶ����ӵ�Ӱ��
    epsilon_Ag = E2r + 1i .* E2i + omega_p .^ 2 ./ ( omega_eV .^ 2 + 1i .* omega_eV .* Gamma_b) - omega_p .^ 2 ./ (omega_eV .^ 2 + 1i .* omega_eV .* Gamma);
    
end

