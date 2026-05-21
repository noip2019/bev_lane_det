import importlib
import os
import site


def _find_real_cv2_package():
    this_dir = os.path.realpath(os.path.dirname(__file__))
    search_roots = []

    try:
        search_roots.extend(site.getsitepackages())
    except AttributeError:
        pass

    user_site = site.getusersitepackages()
    if user_site:
        search_roots.append(user_site)

    for root in search_roots:
        candidate = os.path.join(root, "cv2")
        if os.path.isdir(candidate) and os.path.realpath(candidate) != this_dir:
            return candidate

    raise ImportError("Unable to locate the real OpenCV package")


_REAL_CV2_PACKAGE = _find_real_cv2_package()

if _REAL_CV2_PACKAGE not in __path__:
    __path__.append(_REAL_CV2_PACKAGE)

_native = importlib.import_module(".cv2", __name__)

for name in dir(_native):
    if name.startswith("__") and name != "__version__":
        continue
    globals()[name] = getattr(_native, name)

for module_name in ("data", "gapi", "mat_wrapper", "misc", "utils", "version"):
    try:
        globals()[module_name] = importlib.import_module(f".{module_name}", __name__)
    except ImportError:
        continue

__all__ = [name for name in globals() if not name.startswith("_")]
