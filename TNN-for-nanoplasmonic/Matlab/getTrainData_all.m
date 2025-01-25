%% 变量改变区

%-Lm是指介质层数，包括球壳外；Lf是指定作用介质层；Ls是制定目标介质层。
    Lm = 3; Ls = 1; Lf = 1;

%-壳层介质边界条件
    low_bound_Ag = 1; low_bound_Au2S = 1;
    up_bound_Ag = 5; up_bound_Au2S = 30;

%-量子点与壳层距离边界条件 hd:1-3nm
    hd = 3;

%-数据集大小
    data_size = 10000;
    
%?光谱范围
    omega_eV = ( 1 : 0.00375 : 2.49625 );

%% 定义常量
    
    h = 6.62607015e-34;
    hba = h ./ (2 .* pi);

    eq = 1.60217733e-19; % 单位电荷量

    epsilon_0 = 8.854187817e-12; % 真空介电常数ε0

    c = 2.9979258e8; % 光速

%?光谱单位转换
    omega = omega_eV .* eq ./ hba;

%% 定义介质系数

%-计算 相对电容率 和 相对磁导率 常量  Si02:2.13   Au2S:5.4   Si:11.7
    epsilon_Au2S = 5.4 .* ones(size(omega)); % 在水中的Au2S的相对介电常数
    epsilon_norm = 1.78 .* ones(size(omega)); % 1.78介电常数的水介质
    mu_norm = 1; % 磁导率

%% 数据 生成 和 输出

    data_X = []; data_Y1 = []; data_Y2 = [];
    
    for data_cur = 1 : data_size - 1
        
        if mod(data_cur , 1000) == 0
            strcat('data_cur=', num2str(data_cur))
        end
        
%-计算产生 壳层结构的 半径 和 相对电容率 和 相对磁导率     
        Rd = []; Rf = []; Ri_last_c = 0;
        epsilon = [];
        for Lcur = Lm : -1 : 2
            % 如果Lcur是偶数，生成 Ag 层的半径和相对电容率
            if mod(Lcur, 2) == 0
                Rdi = round(rand * (up_bound_Ag - low_bound_Ag) + low_bound_Ag, 1);
                epsiloni = epseilonAg(omega_eV, Rdi);
            % 如果Lcur是奇数，生成 Au2S 层的半径和相对电容率。
            else
                Rdi = round(rand * (up_bound_Au2S - low_bound_Au2S)+low_bound_Au2S, 1);
                epsiloni = epsilon_Au2S;
            end
            Rd = [Rdi, Rd]; % 单层厚度，数据靠后越靠近核[1.7, 24.1]
            
            Ri = Rdi + Ri_last_c;
            Ri_last_c = Ri;
            Rf = [Ri; Rf]; % 累计厚度 [25.8, 24.1]
            
            epsilon = [epsiloni; epsilon]; % 各层介质点介常数 2*400
            
        end
        
        R = hd + Rf(1, 1); % 模型半径(量子点距球心距离=量子点距球表面+球半径)
        
        Rf = Rf .* 1e-9;
        R = R .* 1e-9;
        
        epsilon = [epsilon_norm; epsilon]; % 三层的介电常数 3*400
        mu = ones(Lm, 1) .* mu_norm; % 磁导率 [1,1,1]

%-计算 波数
        kappa = sqrt(epsilon .* mu) .* (omega ./ c); % k=sqrt(εμ)*(ω/c)
        
%-计算 格林函数 和 局域态密度
        [Gfs1, Fn1] = sphereGreenFunction(R, 0, 0, R,  0, 0, kappa, mu, Rf, Lm, Ls, Lf);
%         [Gfs2, Fn2] = sphereGreenFunction(R, 0, 0, R, pi, 0, kappa, mu, Rf, Lm, Ls, Lf);
%         Fn3 = abs( Fn1 - Fn2 );

%-数据整理      
       data_X(end + 1, :) =  Rd;
       data_Y1(end + 1, :) = Fn1;
       data_Y2(end + 1, :) = Fn3;
       
    end
%% 绘图处理
    figure(); hold;
    plot( omega_eV, Fn1 );
    plot( omega_eV, Fn2 );
    plot( omega_eV, Fn3 );
    
%% 数据文件输出 格式为h5
    
    split_size = round(data_size .* 0.8);
    
    train_X = data_X(1 : split_size, :)';
    valid_X = data_X(split_size : end, :)';
    
    train_Y1 = data_Y1(1 : split_size, :)';
    valid_Y1 = data_Y1(split_size : end, :)';
    
    train_Y2 = data_Y2(1 : split_size, :)';
    valid_Y2 = data_Y2(split_size : end, :)';
    
    cd('..');
    
    h5_FileName = strcat(num2str(Lm-1), '_train_data_Au2S.h5');
    
    if exist(h5_FileName, 'file')
        delete(h5_FileName);
    end
    
    h5create( h5_FileName, '/train_X', size(train_X) );
    h5write( h5_FileName, '/train_X', train_X );
    
    h5create( h5_FileName, '/valid_X', size(valid_X) );    
    h5write( h5_FileName, '/valid_X', valid_X );
    
    h5create( h5_FileName, '/train_Y1', size(train_Y1) );
    h5write( h5_FileName, '/train_Y1', train_Y1 );
    
    h5create( h5_FileName, '/valid_Y1', size(valid_Y1) );
    h5write( h5_FileName, '/valid_Y1', valid_Y1 );
    
    h5create( h5_FileName, '/train_Y2', size(train_Y2) );
    h5write( h5_FileName, '/train_Y2', train_Y2 );
    
    h5create( h5_FileName, '/valid_Y2', size(valid_Y2) );
    h5write( h5_FileName, '/valid_Y2', valid_Y2 );
    
    cd('../Matlab');
    