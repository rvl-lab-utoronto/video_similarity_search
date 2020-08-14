"""
Modified from https://github.com/kenshohara/3D-ResNets-PyTorch
"""
import io
import os
import h5py
from PIL import Image


class ImageLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path,'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class BinaryImageLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        image_file = Image.open(path) # open colour image
        image_file = image_file.convert('L') # convert image to black and white
        return image_file


class ImageLoaderAccImage(object):

    def __call__(self, path):
        import accimage
        return accimage.Image(str(path))


class VideoLoader(object):

    def __init__(self, image_name_formatter, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader()

    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = video_path + '/' + self.image_name_formatter(i)
            if os.path.exists(image_path):
                video.append(self.image_loader(image_path))
            else:
                print('Image DNE:', image_path)

        return video


class VideoLoaderHDF5(object):

    def __call__(self, video_path, frame_indices):
        with h5py.File(video_path, 'r') as f:
            video_data = f['video']

            video = []
            for i in frame_indices:
                if i < len(video_data):
                    video.append(Image.open(io.BytesIO(video_data[i])))
                else:
                    return video

        return video


class VideoLoaderFlowHDF5(object):

    def __init__(self):
        self.flows = ['u', 'v']

    def __call__(self, video_path, frame_indices):
        with h5py.File(video_path, 'r') as f:

            flow_data = []
            for flow in self.flows:
                flow_data.append(f[f'video_{flow}'])

            video = []
            for i in frame_indices:
                if i < len(flow_data[0]):
                    frame = [
                        Image.open(io.BytesIO(video_data[i]))
                        for video_data in flow_data
                    ]
                    frame.append(frame[-1])  # add dummy data into third channel
                    video.append(Image.merge('RGB', frame))

        return video


if __name__ == '__main__':
    import cv2
    loader=BinaryImageLoaderPIL()
    x = loader('/media/diskstation/datasets/UCF101/lw_pose/train/TableTennisShot/v_TableTennisShot_g17_c03/image_00001_kp.png')
    print(x)
    x.save('result.png')
