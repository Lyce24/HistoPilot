# ruff: noqa: F821
"""Exact legacy methods used to exercise decoder ownership without native libraries.

Source: pinned OpenSDPC a07579eedde1dffddf8fa712ef236b97ca8cfc55.
Globals and native calls are supplied by the regression test.
"""


class OldSdpc:
    def __init__(self, sdpcPath):
        self.sdpcPath = sdpcPath
        self.sdpc = self.readSdpc(self.sdpcPath)
        self.level_count = self.getLevelCount()
        self.level_downsamples = self.getLevelDownsamples()
        self.level_dimensions = self.getLevelDimensions()
        self.scan_magnification = self.readSdpc(self.sdpcPath).contents.picHead.contents.rate
        self.sampling_rate = self.readSdpc(self.sdpcPath).contents.picHead.contents.scale

    def read_region(self, location, level, size):
        """Return a PIL.Image containing the contents of the region.
        location: (x, y) tuple giving the top left pixel in the level 0
                  reference frame.
        level:    the level number.
        size:     (width, height) tuple giving the region size."""
        startX, startY = location
        scale = self.level_downsamples[level]
        startX = int(startX / scale)
        startY = int(startY / scale)

        width, height = size

        rgbPos = POINTER(c_uint8)()
        rgbPosPointer = byref(rgbPos)
        so.SqGetRoiRgbOfSpecifyLayer(self.sdpc, rgbPosPointer, width, height, startX, startY, level)
        rgb = self.getRgb(rgbPos, width, height)[..., ::-1]
        rgbCopy = rgb.copy()
        rgbCopy = Image.fromarray(rgbCopy)  # NOTE: new add

        so.Dispose(rgbPos)
        del rgbPos
        del rgbPosPointer
        gc.collect()

        return rgbCopy
