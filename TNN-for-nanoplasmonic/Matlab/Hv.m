function [Y] = Hv(f, s)

    y = (f - s) >= 0;
    Y = double(y);

end

