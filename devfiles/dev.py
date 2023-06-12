import inspect


# def f(a, b, c='c', **kwargs):
#     return a, b, c

class A:
    def __call__(self, a, b, c):
        return a, b, c


# sig = inspect.signature(f)
sig = inspect.signature(A())

def extend_signature(fn):
    """ Decorator to extend the signature of a function, giving the possiblity of 
    receiving with any keyword arguments.
    """
    sig = inspect.signature(fn)

    if inspect._ParameterKind.VAR_KEYWORD in [p.kind for p in sig.parameters.values()]:
        return fn

    def wrapped(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(*args, **kwargs)

    wrapped.__signature__ = sig
    return wrapped


g = extend_signature(A())
print(g(1, 2, c=3, d=5))
