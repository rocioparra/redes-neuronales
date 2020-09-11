from functools import wraps
from os.path import isfile
import pickle
from inspect import signature


def autosave(fmt, extension='p'):
    def decorator_autosave(f):
        @wraps(f)
        def wrapper_autosave(*args, **kwargs):
            params = signature(f).parameters 
            
            # start with all default values (some might be empty)
            args_dict = {arg_name: arg.default for arg_name, arg in params.items()}
        
            # update with received positional arguments
            arg_names = list(params.keys())
            args_dict.update({arg_names[i]: arg_value for i, arg_value in enumerate(args)})
            
            # update with received keyword arguments
            args_dict.update(kwargs)
            
            # get file name using all required arguments 
            filename = f'{fmt.format(**args_dict)}.{extension}'

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


def print_list(l):
    print(f"""['{r"', '".join(l)}']""")
    print(f"Palabras totales: {len(l)}")
    print(f"Palabras distintas: {len(set(l))}")
