import torch
import cv2
import numpy as np

'''
# Functions for visualization
def vid_tensor_to_numpy(vid, is_batch=False):
    # channels x frames x width, height --> frames x width x height x channels
    vid_np = vid
    if is_batch:
        vid_np = vid[0]
    vid_np = vid_np.permute(1,2,3,0).numpy()
    return vid_np

def cv_f32_to_u8 (img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = np.uint8(255 * img)
    return img
'''

def construct_net_input(vid_loader, channel_ext, spatial_transform,
        normalize_fn, path, frame_indices, channel_paths={},
        pos_channel_replace=False, prob_pos_channel_replace=None):

    if prob_pos_channel_replace is None:
        prob_pos_channel_replace = 0.25  # default val

    clip = vid_loader(path, frame_indices)

    if spatial_transform is not None:
        spatial_transform.randomize_parameters()
        clip = [spatial_transform(img) for img in clip]

    if pos_channel_replace:

        # Threshold for mean for salient view (if > threshold, salient view is
        # not empty, replace)
        SALIENT_MASK_THRESHOLD = 0.01

        #Fixed % chance to replace rgb positive with another view
        choices = ['replace', 'rgb']
        choice = np.random.choice(choices, p=[prob_pos_channel_replace,
                                              1.0-prob_pos_channel_replace])

        if choice == 'replace':
            assert len(channel_paths) == 1, 'Only 1 other view for now'
            for key_i in channel_paths:
                key = key_i
                break
            channel_path = channel_paths[key]
            channel_loader = channel_ext[key][1]
            channel_clip = channel_loader(channel_path, frame_indices)
            if spatial_transform is not None:
                channel_clip = [spatial_transform(img) for img in channel_clip]

            if key != 'salient' or key == 'salient' and \
                    torch.mean(torch.stack(channel_clip, 0)) >= SALIENT_MASK_THRESHOLD:

                clip = [torch.cat((channel_clip[i], channel_clip[i], channel_clip[i]), dim=0) for i in
                    range(len(channel_clip))]
                #print('replaced clip')
            #else:
                #print('didnt replace clip')
            #print('clip dim', clip[0].shape)

    else:
        for key in channel_paths:
            channel_path = channel_paths[key]
            channel_loader = channel_ext[key][1]

            channel_clip = channel_loader(channel_path, frame_indices)
            if spatial_transform is not None:
                channel_clip = [spatial_transform(img) for img in channel_clip]
            clip = [torch.cat((clip[i], channel_clip[i]), dim=0) for i in range(len(clip))]

    clip = [normalize_fn(img) for img in clip]
    clip = torch.stack(clip, 0).permute(1, 0, 2, 3) #change to (C, D, H, W)

    '''
    # Visualization - for debugging
    center_img_salient = clip[:,16//2, :,:]
    center_img = vid_tensor_to_numpy(center_img_salient.unsqueeze(1))[0]
    center_img = cv2.cvtColor(center_img, cv2.COLOR_RGB2BGR)
    center_img = cv_f32_to_u8(center_img)
    cv2.imshow('input', center_img)
    cv2.waitKey()'''
    return clip
