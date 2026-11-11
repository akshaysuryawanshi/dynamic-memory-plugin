"""Small transport-independent errors for the resize controller."""

MAX_TARGET_BYTES = 2**63 - 1


class ControlError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def as_dict(self):
        return {"code": self.code, "message": str(self)}
