class RealAlphaFoldAdapter:
    def __init__(self, protocol):
        if protocol.scope != "CALIBRATED_EXTERNAL":
            raise ValueError("BLOCKED real backend/version/weights/calibration")
        raise NotImplementedError("G3/G6 oracle approval required; no hidden downloads/MSA calls")
