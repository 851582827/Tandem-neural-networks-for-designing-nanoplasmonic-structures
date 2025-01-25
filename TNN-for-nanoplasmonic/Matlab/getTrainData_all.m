%% �����ı���

%-Lm��ָ���ʲ�������������⣻Lf��ָ�����ý��ʲ㣻Ls���ƶ�Ŀ����ʲ㡣
    Lm = 3; Ls = 1; Lf = 1;

%-�ǲ���ʱ߽�����
    low_bound_Ag = 1; low_bound_Au2S = 1;
    up_bound_Ag = 5; up_bound_Au2S = 30;

%-���ӵ���ǲ����߽����� hd:1-3nm
    hd = 3;

%-���ݼ���С
    data_size = 10000;
    
%?���׷�Χ
    omega_eV = ( 1 : 0.00375 : 2.49625 );

%% ���峣��
    
    h = 6.62607015e-34;
    hba = h ./ (2 .* pi);

    eq = 1.60217733e-19; % ��λ�����

    epsilon_0 = 8.854187817e-12; % ��ս�糣����0

    c = 2.9979258e8; % ����

%?���׵�λת��
    omega = omega_eV .* eq ./ hba;

%% �������ϵ��

%-���� ��Ե����� �� ��Դŵ��� ����  Si02:2.13   Au2S:5.4   Si:11.7
    epsilon_Au2S = 5.4 .* ones(size(omega)); % ��ˮ�е�Au2S����Խ�糣��
    epsilon_norm = 1.78 .* ones(size(omega)); % 1.78��糣����ˮ����
    mu_norm = 1; % �ŵ���

%% ���� ���� �� ���

    data_X = []; data_Y1 = []; data_Y2 = [];
    
    for data_cur = 1 : data_size - 1
        
        if mod(data_cur , 1000) == 0
            strcat('data_cur=', num2str(data_cur))
        end
        
%-������� �ǲ�ṹ�� �뾶 �� ��Ե����� �� ��Դŵ���     
        Rd = []; Rf = []; Ri_last_c = 0;
        epsilon = [];
        for Lcur = Lm : -1 : 2
            % ���Lcur��ż�������� Ag ��İ뾶����Ե�����
            if mod(Lcur, 2) == 0
                Rdi = round(rand * (up_bound_Ag - low_bound_Ag) + low_bound_Ag, 1);
                epsiloni = epseilonAg(omega_eV, Rdi);
            % ���Lcur������������ Au2S ��İ뾶����Ե����ʡ�
            else
                Rdi = round(rand * (up_bound_Au2S - low_bound_Au2S)+low_bound_Au2S, 1);
                epsiloni = epsilon_Au2S;
            end
            Rd = [Rdi, Rd]; % �����ȣ����ݿ���Խ������[1.7, 24.1]
            
            Ri = Rdi + Ri_last_c;
            Ri_last_c = Ri;
            Rf = [Ri; Rf]; % �ۼƺ�� [25.8, 24.1]
            
            epsilon = [epsiloni; epsilon]; % ������ʵ�鳣�� 2*400
            
        end
        
        R = hd + Rf(1, 1); % ģ�Ͱ뾶(���ӵ�����ľ���=���ӵ�������+��뾶)
        
        Rf = Rf .* 1e-9;
        R = R .* 1e-9;
        
        epsilon = [epsilon_norm; epsilon]; % ����Ľ�糣�� 3*400
        mu = ones(Lm, 1) .* mu_norm; % �ŵ��� [1,1,1]

%-���� ����
        kappa = sqrt(epsilon .* mu) .* (omega ./ c); % k=sqrt(�Ŧ�)*(��/c)
        
%-���� ���ֺ��� �� ����̬�ܶ�
        [Gfs1, Fn1] = sphereGreenFunction(R, 0, 0, R,  0, 0, kappa, mu, Rf, Lm, Ls, Lf);
%         [Gfs2, Fn2] = sphereGreenFunction(R, 0, 0, R, pi, 0, kappa, mu, Rf, Lm, Ls, Lf);
%         Fn3 = abs( Fn1 - Fn2 );

%-��������      
       data_X(end + 1, :) =  Rd;
       data_Y1(end + 1, :) = Fn1;
       data_Y2(end + 1, :) = Fn3;
       
    end
%% ��ͼ����
    figure(); hold;
    plot( omega_eV, Fn1 );
    plot( omega_eV, Fn2 );
    plot( omega_eV, Fn3 );
    
%% �����ļ���� ��ʽΪh5
    
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
    