class AbortExperimentError(Exception):
    """
    Custom exception class for the training pipeline.

    Raised simultaneously by all subprocesses whenever the rank 0 subprocess fails to perform a
    task while all other subprocesses are waiting.

    :param message: str
        Error message.
    :param orig_traceback: any
        Traceback of the originally raised exception in the rank 0 subprocess.
    """

    def __init__(
        self,
        message,
        orig_traceback
    ):
    
        super().__init__(message)
        self.orig_traceback = orig_traceback
