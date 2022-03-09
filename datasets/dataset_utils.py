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

def get_channel_clip(channel_ext, channel_paths, frame_indices,
        spatial_transform, key):

    U_ONLY = True
    channel_loader = channel_ext[key][1]

    if U_ONLY:
        channel_path = channel_paths[key]
        channel_clip = channel_loader(channel_path, frame_indices)
        if spatial_transform is not None:
            channel_clip = [spatial_transform(img) for img in channel_clip]

        if key != 'salient' or key == 'salient' and \
                torch.mean(torch.stack(channel_clip, 0)) >= SALIENT_MASK_THRESHOLD:

            channel_clip = [torch.cat((channel_clip[i], channel_clip[i], channel_clip[i]), dim=0) for i in
                range(len(channel_clip))]

        else:
            assert False, 'not supported'

    else:
        channel_path_u = channel_paths[key]
        channel_path_v = channel_path_u.replace("/u/", "/v/")

        channel_clip_u = channel_loader(channel_path_u, frame_indices)
        channel_clip_v = channel_loader(channel_path_v, frame_indices)

        if spatial_transform is not None:
            channel_clip_u = [spatial_transform(img) for img in channel_clip_u]
            channel_clip_v = [spatial_transform(img) for img in channel_clip_v]

        #if key != 'salient' or key == 'salient' and \
        #        torch.mean(torch.stack(channel_clip, 0)) >= SALIENT_MASK_THRESHOLD:

        channel_clip = [torch.cat((channel_clip_u[i], channel_clip_v[i],
                        torch.zeros(channel_clip_u[i].shape)), dim=0) for i in range(len(channel_clip_u))]

    return channel_clip


def construct_net_input(vid_loader, channel_ext, spatial_transform,
        normalize_fn, path, frame_indices, channel_paths={},
        pos_channel_replace=False, prob_pos_channel_replace=None,
        modality=False, split='train', flow_only=False):

    assert not (split != 'train' and pos_channel_replace)

    if prob_pos_channel_replace is None:
        prob_pos_channel_replace = 0.25  # default val

    if flow_only:
        assert not modality and not pos_channel_replace
        assert len(channel_paths) >= 1, 'the channel path is empty!'
        key = np.random.choice(list(channel_paths)) #randomly select a view as positive
        clip = get_channel_clip(channel_ext, channel_paths,
                    frame_indices, spatial_transform, key)

    else:
        clip = vid_loader(path, frame_indices)

        if spatial_transform is not None:
            spatial_transform.randomize_parameters()
            clip = [spatial_transform(img) for img in clip]

    SALIENT_MASK_THRESHOLD = 0.01

    if modality:
        assert len(channel_paths) == 1, 'Only 1 other view for now'
        for key_i in channel_paths:
            key = key_i
            break
        channel_path = channel_paths[key]
        channel_loader = channel_ext[key][1]
        print(channel_loader)
        channel_clip = channel_loader(channel_path, frame_indices)
        if spatial_transform is not None:
            channel_clip = [spatial_transform(img) for img in channel_clip]

        if key != 'salient' or key == 'salient' and \
                torch.mean(torch.stack(channel_clip, 0)) >= SALIENT_MASK_THRESHOLD:

            channel_clip = [torch.cat((channel_clip[i], channel_clip[i], channel_clip[i]), dim=0) for i in
                range(len(channel_clip))]
        else:
            channel_clip = clip

        clip = [normalize_fn(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3) #change to (C, D, H, W)
        channel_clip = [normalize_fn(img) for img in channel_clip]
        channel_clip= torch.stack(channel_clip, 0).permute(1, 0, 2, 3) #change to (C, D, H, W)
        return [clip, channel_clip]

    elif pos_channel_replace:
        # Threshold for mean for salient view (if > threshold, salient view is
        # not empty, replace)
        #Fixed % chance to replace rgb positive with another view
        choices = ['replace', 'rgb']
        choice = np.random.choice(choices, p=[prob_pos_channel_replace,
                                              1.0-prob_pos_channel_replace])

        if choice == 'replace':
            assert len(channel_paths) >= 1, 'the channel path is empty!'
            key = np.random.choice(list(channel_paths)) #randomly select a view as positive
            # print('choice repalce, key:', key)
            # if len(channel_paths) ==1:
            #     for key_i in channel_paths:
            #         key = key_i
            #         break

            clip = get_channel_clip(channel_ext, channel_paths,
                    frame_indices, spatial_transform, key)

    elif not flow_only:
        for key in channel_paths:
            channel_path = channel_paths[key]
            channel_loader = channel_ext[key][1]

            channel_clip = channel_loader(channel_path, frame_indices)
            if spatial_transform is not None:
                channel_clip = [spatial_transform(img) for img in channel_clip]
            clip = [torch.cat((clip[i], channel_clip[i]), dim=0) for i in range(len(clip))]
    clip = [normalize_fn(img) for img in clip]
    if len(clip) == 0:
        print('Clip with size 0:', path)
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
