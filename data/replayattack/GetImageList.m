clc; clear all; close all;

phase = 'test';
face_data_dir = sprintf('%s_face_data', phase);
out_image_list = sprintf('%s_image_list.txt', phase);
out_label_list   = sprintf('%s_label_list.txt', phase);

%% list for real images
real_img_path = fullfile('.', face_data_dir, 'real');
real_img_path_list = {};
video_list = dir(real_img_path);
for video_ind = 3 : length(video_list)
    video_name = video_list(video_ind).name;
    frame_list = dir(fullfile(real_img_path, video_name));
    for frame_ind = 3 : length(frame_list)
        frame_name = frame_list(frame_ind).name;
        frame_path = fullfile(real_img_path, video_name, frame_name);
        real_img_path_list = [real_img_path_list; frame_path];
    end
end

%% list for attack images
% hand hold
hand_img_path = fullfile('.', face_data_dir, 'attack/hand');
hand_img_path_list = {};
video_list = dir(hand_img_path);
for video_ind = 3 : length(video_list)
    video_name = video_list(video_ind).name;
    frame_list = dir(fullfile(hand_img_path, video_name));
    for frame_ind = 3 : length(frame_list)
        frame_name = frame_list(frame_ind).name;
        frame_path = fullfile(hand_img_path, video_name, frame_name);
        hand_img_path_list = [hand_img_path_list; frame_path];
    end
end

% fixed
fixed_img_path = fullfile('.', face_data_dir, 'attack/fixed');
fixed_img_path_list = {};
video_list = dir(fixed_img_path);
for video_ind = 3 : length(video_list)
    video_name = video_list(video_ind).name;
    frame_list = dir(fullfile(fixed_img_path, video_name));
    for frame_ind = 3 : length(frame_list)
        frame_name = frame_list(frame_ind).name;
        frame_path = fullfile(fixed_img_path, video_name, frame_name);
        fixed_img_path_list = [fixed_img_path_list; frame_path];
    end
end

%% Write the list into files    
img_path_list = [real_img_path_list; hand_img_path_list; fixed_img_path_list];
label_list = [ones(length(real_img_path_list), 1); zeros(length(hand_img_path_list) + length(fixed_img_path_list), 1)];

img_list_writer = fopen(fullfile('.', sprintf('%s_image_list.txt', phase)), 'w');
label_writer = fopen(fullfile('.', sprintf('%s_label_list.txt', phase)), 'w');
for i = 1 : length(img_path_list)
    fprintf(img_list_writer, '%s\n', img_path_list{i});
    fprintf(label_writer, '%d\n', label_list(i));
end
fclose(img_list_writer);
fclose(label_writer);