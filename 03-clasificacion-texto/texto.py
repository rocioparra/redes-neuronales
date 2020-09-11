from functools import wraps
from os.path import isfile
import pickle
from inspect import signature


def autosave(fmt):
    def decorator_autosave(f):
        @wraps(f)
        def wrapper_autosave(*args, **kwargs):
            params = signature(f).parameters
            args_dict = {arg_name: arg.default for arg_name, arg in params.items()}

            arg_names = list(params.keys())
            args_dict.update({arg_names[i]: arg_value for i, arg_value in enumerate(args)})
            args_dict.update(kwargs)
            filename = fmt.format(**args_dict)

            if isfile(filename):
                with open(filename, 'rb') as fp:
                    result = pickle.load(fp)
            else:
                result = f(*args, **kwargs)
                with open(filename, 'wb') as fp:
                    pickle.dump(result, fp)
            return result
        return wrapper_autosave
    return decorator_autosave


@autosave(fmt='facto-{n}.p')
def factorial(n):
    f = 1
    for i in range(2, n+1):
        f *= i
    return f


print(factorial(8))
print(factorial(n=4))
print(factorial(4))
print(factorial(9))
