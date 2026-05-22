class _Logger:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass
    def bind(self, *args, **kwargs): return self

def get_logger():
    return _Logger()
