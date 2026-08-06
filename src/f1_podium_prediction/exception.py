class F1PipelineError(Exception):
    pass

class DataValidationError(F1PipelineError):
    pass


class ModelNotFoundError(F1PipelineError):
    pass