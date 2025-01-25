%% 变量改变区自发辐射

%-Lm是指介质层数，包括球壳外；Lf是指定作用介质层；Ls是制定目标介质层。
    Lm = 3; Ls = 1; Lf = 1;
    
%-保存符号
    save_sym = '1m1';

%-设置角坐标
    Theta1 = 0;
    
%-量子点与壳层距离边界条件
    hd_1 = 1.55;
%-光谱范围
    omega_eV = ( 1 : 0.00375 : 2.49625 );

%% 四层边界条件

%-比较结果
%     Rd = [1.9, 18.4];
%    
%-优化结果
%   Au2S
    Rd = [2.62, 15.35];  
%   Si
%     Rd = [2.61, 9.03];  
%   SiO2
%     Rd = [2.1, 19.3];  

%% 其他边界条件

%     Rd = [01.35, 07.37, 01.69, 04.13];
%     Rd = [01.06, 07.85, 04.02, 07.30];
%     Rd = [01.09, 21.15, 01.97, 05.66];
%     Rd = [01.22, 11.00, 02.12, 04.40];
    
%% 定义常量
    
    h = 6.62607015e-34;
    hba = h ./ (2 .* pi);

    eq = 1.60217733e-19;

    epsilon_0 = 8.854187817e-12;

    c = 2.9979258e8;

%-光谱单位转换
    omega = omega_eV .* eq ./ hba;

%% 定义介质系数

%-计算 相对电容率 和 相对磁导率 常量
%-Au2S
    epsilon_Au2S = 5.4 .* ones(size(omega));
%-Si
%     epsilon_Au2S = 11.7 .* ones(size(omega));
%-SiO2
%     epsilon_Au2S = 2.13 .* ones(size(omega));

    epsilon_norm = 1.78 .* ones(size(omega));
    mu_norm = 1;

%% 数据 生成 和 输出

    Rf = []; Ri_last_c = 0;
    epsilon = [];
    
    Gfs_all = []; Fn_all = [];
    
    for Lcur = Lm - 1 : -1 : 1
        Rdi = Rd(Lcur);
        if mod(Lcur, 2) == 1
            epsiloni = epseilonAg(omega_eV, Rdi);
        else
            epsiloni = epsilon_Au2S;
        end
        Ri = Rdi + Ri_last_c;
        Ri_last_c = Ri;
        Rf = [Ri; Rf];
            
        epsilon = [epsiloni; epsilon];
        
    end
    R1 = hd_1 + Rf(1, 1);
    Rf = Rf .* 1e-9;
    R1 = R1 .* 1e-9;
    
    epsilon = [epsilon_norm; epsilon];
    mu = ones(Lm, 1) .* mu_norm;
    
    %-计算 波数
    kappa = sqrt(epsilon .* mu) .* (omega ./ c);
    
    %-计算 格林函数 和 局域态密度

    Theta_1 = Theta1;
    Theta_2 = Theta1;
    R_1 = R1;
    R_2 = R1;

    [Gfs, Fn] = sphereGreenFunction(R_1, Theta_1, 0, R_2, Theta_2, 0, kappa, mu, Rf, Lm, Ls, Lf);

    Gfs_all = Gfs;
    Fn_all = Fn;

%% 计算机率

%-D2是偶极矩的平方
    D2 = (60 .* 3.334 .* 10 .^ -30) .^ 2;
    
%-Gfs_all_cf 为 - Delta + i Gamma / 2     
    Gfs_all_cf = D2 .* Gfs_all .* ((omega ./ c) .^ 2) ./hba ./ epsilon_0;  
    
    C_a_0 = 1;
    
    omega_a0_eV = 1.6;
    omega_ad = real(Gfs_all_cf(1, omega_eV == omega_a0_eV));
    omega_a = omega_a0_eV .* eq ./ hba + omega_ad;
     
%-计算 Ca   
    C_a_w_cf_p = (omega - omega_a + Gfs_all_cf(1, :));
    C_a_w_p = 1i .* C_a_0 ./ (C_a_w_cf_p);
   
%-计算 C_a_w 和 C_a_t
    d_omega = 0.0000375 .* eq ./ hba;
    d_t = 1./ 3e14;
    N = 500;
    
    C_a_w = 2 .* real(C_a_w_p);
    
    [ C_a_t, x_t_a ] = IFFT( C_a_w, d_omega, omega, d_t, N );

%% 绘图
    
    figure(); hold;
    plot( omega_eV, Fn_all( 1, : ) );
    
    figure(); hold;
    plot( omega_eV, C_a_w );
    
    figure(); hold;
    plot( x_t_a, abs(C_a_t) );
    set(gca, 'YTick', [  0 : 0.1 : 1.01]);
    set(gca, 'Ygrid', 'on');
    
%% 数据文件输出 格式为h5
    
    struct_X = Rd';
    ld_Y = Fn_all( 1, : )';
    frec_X = omega_eV';
    frec_Y = C_a_w';
    time_X = x_t_a';
    time_Y1 = abs(C_a_t)';
    
    cd('../new_design');
    % Au2S Si SiO2
    h5_FileName = strcat(num2str(Lm-1), '_coupling_data_', save_sym, '_SiO2_youhua','.h5');
    
    if exist(h5_FileName, 'file')
        delete(h5_FileName);
    end
    
    h5create( h5_FileName, '/struct_X', size(struct_X) );
    h5write( h5_FileName, '/struct_X', struct_X );
    
    h5create( h5_FileName, '/ld_Y', size(ld_Y) );
    h5write( h5_FileName, '/ld_Y', ld_Y );
    
    h5create( h5_FileName, '/frec_X', size(frec_X) );
    h5write( h5_FileName, '/frec_X', frec_X );
    
    h5create( h5_FileName, '/frec_Y', size(frec_Y) );
    h5write( h5_FileName, '/frec_Y', frec_Y );
    
    h5create( h5_FileName, '/time_X', size(time_X) );
    h5write( h5_FileName, '/time_X', time_X );
    
    h5create( h5_FileName, '/time_Y1', size(time_Y1) );
    h5write( h5_FileName, '/time_Y1', time_Y1 );
    
    cd('../Matlab');