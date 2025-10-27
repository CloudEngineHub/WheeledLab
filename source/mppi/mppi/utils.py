import importlib

# Resolve classes using module if not already a class type
def resolve_obj(class_key: str) -> type:
    """
    Resolve a class from a configuration dictionary.

    Args:
        cfg (dict): Configuration dictionary.
        class_key (str): Key in the configuration dictionary that contains the class name.
        module (str): Module name where the class is located.

    Returns:
        type: Resolved class type.
    """
    if isinstance(class_key, str):
        return resolve_from_str(class_key)

def resolve_from_str(full_path: str):
    """
    Resolves a class or function from a module using a string in the format
    'module.submodule...finalmodule:class_or_func_name' and instantiates or calls it.

    Args:
        full_path (str): The full path string (e.g., 'module.submodule...finalmodule:class_or_func_name').
        *args: Positional arguments to pass to the class constructor or function.
        **kwargs: Keyword arguments to pass to the class constructor or function.

    Returns:
        object: The instantiated class or the result of the function call.
    """
    # Split the string into module path and class/function name
    module_path, class_or_func_name = full_path.split(":")

    # Import the module dynamically
    module = importlib.import_module(module_path)

    # Resolve the class or function
    resolved = getattr(module, class_or_func_name)

    # If it's a class, instantiate it; if it's a function, call it
    if callable(resolved):
        return resolved
    else:
        raise TypeError(f"{class_or_func_name} is not callable in module {module_path}")