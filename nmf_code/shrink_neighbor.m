function img=shrink_neighbor(weight)
img = zeros(5,5);
for oi = 1:3
    for oj = 1:3
        for ii = 1:3
            for ij = 1:3
                img(2 + (oi - 1) + (ii - 2),2 + (oj - 1) + (ij - 2)) = ...
                    img(2 + (oi - 1) + (ii - 2),2 + (oj - 1) + (ij - 2)) + weight(oi,oj,ii,ij);
            end
        end
    end
end
end