try:
    from gfpgan import GFPGANer
except ModuleNotFoundError:
    GFPGANer = None

class FaceEnhancer:
    def __init__(self, model_path):
        if GFPGANer is None:
            self.enabled = False
            print("[INFO] GFPGAN not installed. Enhancement disabled.")
            return
        self.enabled = True
        self.enhancer = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )

    def enhance(self, face_img):
        if not self.enabled:
            return face_img
        _, _, enhanced = self.enhancer.enhance(
            face_img,
            has_aligned=False,
            only_center_face=True
        )
        return enhanced
