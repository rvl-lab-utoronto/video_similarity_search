import torch


def construct_net_input(vid_loader, channel_ext, spatial_transform, normalize_fn, path, frame_indices, channel_paths={}):
    clip = vid_loader(path, frame_indices)

    if spatial_transform is not None:
        spatial_transform.randomize_parameters()
        clip = [spatial_transform(img) for img in clip]

    for key in channel_paths:
        channel_path = channel_paths[key]
        channel_loader = channel_ext[key][1]

        channel_clip = channel_loader(channel_path, frame_indices)
        if spatial_transform is not None:
            channel_clip = [spatial_transform(img) for img in channel_clip]
        clip = [torch.cat((clip[i], channel_clip[i]), dim=0) for i in range(len(clip))]

    clip = [normalize_fn(img) for img in clip]
    clip = torch.stack(clip, 0).permute(1, 0, 2, 3) #change to (C, D, H, W)
    return clip