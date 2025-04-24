import cv2

from layers import BaseShader, Identity

class Webcam(Identity):
    """
    Webcam layer for capturing images from a webcam.
    """

    def __init__(self, ctx, logger, resolution):
        super().__init__(ctx, logger)
        self.webcam = cv2.VideoCapture(0)

        self.texture = ctx.texture(resolution, 3)
        self.sampler = ctx.sampler(texture=self.texture)

    def render(self, resolution=None, **kwargs):
        ret, frame = self.webcam.read()
        frame = cv2.resize(frame, resolution)
        self.texture.write(frame.tobytes())
        self.sampler.use(location=0)
        kwargs = {'input_texture': 0, 'resolution': resolution}
        super().render(**kwargs)