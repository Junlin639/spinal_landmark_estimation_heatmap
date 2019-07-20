function cobbs = angleCal(coords, Hs, Ws)
    ap_num = 68;
    isshow = false;
    
    cobbs = zeros(size(coords,1),3);
    for i=1:size(coords,1)
        coord = coords(i,:);
        H = Hs(1,i);
        W = Ws(1,i);
        landmarks_ap = [];
        cobb = [];

        p2 = [coord(1:ap_num) ; coord(ap_num+1:ap_num*2)]';
        vnum = ap_num / 4;
        landmarks_ap = [landmarks_ap ; coord(1:ap_num), coord(ap_num+1:ap_num*2)]; %scale landmark coordinates

        cob_angles = zeros(1,3);

        mid_p_v = zeros(size(p2,1)/2,2);
        for n=1:size(p2,1)/2
            mid_p_v(n,:) = (p2(n*2,:) + p2((n-1)*2+1,:))/2;
        end

        %calculate the middle vectors & plot the labeling lines
        mid_p = zeros(size(p2,1)/2,2);
        for n=1:size(p2,1)/4
            mid_p((n-1)*2+1,:) = (p2(n*4-1,:) + p2((n-1)*4+1,:))/2;
            mid_p(n*2,:) = (p2(n*4,:) + p2((n-1)*4+2,:))/2;
        end

        vec_m = zeros(size(mid_p,1)/2,2);

        for n=1:size(mid_p,1)/2
            vec_m(n,:) = mid_p(n*2,:) - mid_p((n-1)*2+1,:);
        end

        mod_v = power(sum(vec_m .* vec_m, 2),0.5);
        dot_v = vec_m * vec_m';

        %calculate the Cobb angle
        angles = acos(roundn(dot_v./(mod_v * mod_v'),-8));
        [maxt, pos1] = max(angles);
        [pt, pos2] = max(maxt);
        pt = pt/pi*180;
        cob_angles(1) = pt;

        if ~isS(mid_p_v) % 'S'

            mod_v1 = power(sum(vec_m(1,:) .* vec_m(1,:), 2),0.5);
            mod_vs1 = power(sum(vec_m(pos2,:) .* vec_m(pos2,:), 2),0.5);
            mod_v2 = power(sum(vec_m(vnum,:) .* vec_m(vnum,:), 2),0.5);
            mod_vs2 = power(sum(vec_m(pos1(pos2),:) .* vec_m(pos1(pos2),:), 2),0.5);

            dot_v1 = vec_m(1,:) * vec_m(pos2,:)';
            dot_v2 = vec_m(vnum,:) * vec_m(pos1(pos2),:)';

            mt = acos(roundn(dot_v1./(mod_v1 * mod_vs1'),-8));
            tl = acos(roundn(dot_v2./(mod_v2 * mod_vs2'),-8));

            mt = mt/pi*180;
            cob_angles(2) = mt;
            tl = tl/pi*180;
            cob_angles(3) = tl;

        else

        % max angle in the upper part
            if (mid_p_v(pos2*2,2) + mid_p_v(pos1(pos2)*2,2)) < H

                %calculate the Cobb angle (upside)
                mod_v_p = power(sum(vec_m(pos2,:) .* vec_m(pos2,:), 2),0.5);
                mod_v1 = power(sum(vec_m(1:pos2,:) .* vec_m(1:pos2,:), 2),0.5);
                dot_v1 = vec_m(pos2,:) * vec_m(1:pos2,:)';

                angles1 = acos(roundn(dot_v1./(mod_v_p * mod_v1'),-8));
                [CobbAn1, pos1_1] = max(angles1);
                mt = CobbAn1/pi*180;
                cob_angles(2) = mt;

                %calculate the Cobb angle?downside?
                mod_v_p2 = power(sum(vec_m(pos1(pos2),:) .* vec_m(pos1(pos2),:), 2),0.5);
                mod_v2 = power(sum(vec_m(pos1(pos2):vnum,:) .* vec_m(pos1(pos2):vnum,:), 2),0.5);
                dot_v2 = vec_m(pos1(pos2),:) * vec_m(pos1(pos2):vnum,:)';

                angles2 = acos(roundn(dot_v2./(mod_v_p2 * mod_v2'),-8));
                [CobbAn2, pos1_2] = max(angles2);
                tl = CobbAn2/pi*180;
                cob_angles(3) = tl;

                pos1_2 = pos1_2 + pos1(pos2) - 1;

            else
                %calculate the Cobb angle (upside)
                mod_v_p = power(sum(vec_m(pos2,:) .* vec_m(pos2,:), 2),0.5);
                mod_v1 = power(sum(vec_m(1:pos2,:) .* vec_m(1:pos2,:), 2),0.5);
                dot_v1 = vec_m(pos2,:) * vec_m(1:pos2,:)';


                angles1 = acos(roundn(dot_v1./(mod_v_p * mod_v1'),-8));
                [CobbAn1, pos1_1] = max(angles1);
                mt = CobbAn1/pi*180;
                cob_angles(2) = mt;

                %calculate the Cobb angle (upper upside)
                mod_v_p2 = power(sum(vec_m(pos1_1,:) .* vec_m(pos1_1,:), 2),0.5);
                mod_v2 = power(sum(vec_m(1:pos1_1,:) .* vec_m(1:pos1_1,:), 2),0.5);
                dot_v2 = vec_m(pos1_1,:) * vec_m(1:pos1_1,:)';

                angles2 = acos(roundn(dot_v2./(mod_v_p2 * mod_v2'),-8));
                [CobbAn2, pos1_2] = max(angles2);
                tl = CobbAn2/pi*180;
                cob_angles(3) = tl;

            end
        end

        output = [ 'the Cobb Angles(PT, MT, TL/L) are '  num2str(pt) ', ' num2str(mt) ' and '  num2str(tl) ...
            ', and the two most tilted vertebrae are ' num2str(pos2) ' and ' num2str(pos1(pos2)) '.\n'];

        fprintf(output);

        cobb = [cobb ; cob_angles]; %cobb angles
        cobbs(i,:) = cobb;
    end
end

function [flag] = isS(p)
    ll = linefun(p);
    flag = sum(sum(ll*ll')) ~= sum(sum(abs(ll*ll')));
end

function [ll] = linefun(p)
    num = size(p,1);
    ll = zeros(num-2,1);
    for i=1:(num-2)
        ll(i) = (p(i,2)-p(num,2))/(p(1,2)-p(num,2)) - (p(i,1)-p(num,1))/(p(1,1)-p(num,1));
    end
end




