imgs = dir('*.jpg');
img_re = [];

i_size = [128, 128];

k = [16, 64, 256];

% % separate
% for i = 1:length(imgs)
%     img = imread(imgs(i).name);
%     img = imresize(img, i_size);
%     img = im2double(img);
%     
%     img_array = [];
%     for j =1:3
%         img_temp = img(:,:,j);
%         img_array = [img_array, img_temp(:)];
%     end
%     
%     idx_array = [];
%     C_array = [];
%     for p = 1:length(k)
%         [idx, C] = kmeans(img_array, k(p));
%         idx_array = [idx_array, idx];
%         C_array = [C_array; C];
%     end
%     
%     imwrite(img, [imgs(i).name(1:end-3),'png']);
%     csvwrite([imgs(i).name(1:end-4),'_array.csv'], img_array);
%     csvwrite([imgs(i).name(1:end-4),'_idx.csv'], idx_array);
%     csvwrite([imgs(i).name(1:end-4),'_C.csv'], C_array);
% end

% all
img_array_all = [];
for i = 1:length(imgs)
    img = imread(imgs(i).name);
    img = imresize(img, i_size);
    img = im2double(img);
    
    img_array = [];
    for j =1:3
        img_temp = img(:,:,j);
        img_array = [img_array, img_temp(:)];
    end
    img_array_all = [img_array_all; img_array];
end

idx_array = [];
C_array = [];
for p = 1:length(k)
    [idx, C] = kmeans(img_array_all, k(p), 'MaxIter', 500);
    idx_array = [idx_array, idx];
    C_array = [C_array; C];
end

csvwrite('all_array.csv', img_array_all);
csvwrite('all_idx.csv', idx_array);
csvwrite('all_C.csv', C_array);
