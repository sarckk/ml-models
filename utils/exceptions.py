class InvalidArgError(ValueError):
    def __init__(self, msg):
        super.__init__(msg)

class NotFittedError(ValueError):
    def __init__(self, msg):
        super.__init__(msg)