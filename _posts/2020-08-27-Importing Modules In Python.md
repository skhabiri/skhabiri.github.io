---
layout: post
title: Importing Modules In Python
subtitle: How to make sure our import always works
gh-repo: skhabiri/FastAPI-Spotify
gh-badge: [star, fork, follow]
tags: [Python, Python Package, Import Module]
image: /assets/img/post4/post4_logo.png
comments: false
---

### How python searches for a file:
When importing from a python file (module or script), python does not use the path that is in __file__, but it uses the file full name and the sys.path to identify the file. The full name is `__package__ + __name__`. If __package__ is None, it’s simply __name__.

### Project Structure:
For example, let’s consider [this project](https://github.com/skhabiri/FastAPI-Spotify). To get the file structure of the project, use: `tree -aI "*\.pyc|__pycache*|\.[^e][^n][^v]*" -o tree.text`.

```bash
FastAPI-Spotify (project)
├── .env (sets environment)
├── LICENSE
├── Pipfile (package dependency)
├── Pipfile.lock
├── Procfile
├── README.md
├── appdir (web app top package)
│   ├── __init__.py (app point of entry)
│   ├── api (sub package)
│   │   ├── BW_Spotify_Final.csv
│   │   ├── BW_Spotify_Final.joblib
│   │   ├── __init__.py
│   │   ├── fedata.py
│   │   ├── ormdb.py
│   │   ├── parser.py
│   │   ├── predict.py
│   │   ├── settings.py
│   │   └── viz.py
│   ├── main.py (runs as top module by the app)
│   └── tests (test package)
│       ├── __init__.py
│       ├── test_main.py
│       ├── test_predict.py
│       └── test_viz.py
├── etc
│   └── new_df.csv
├── notebooks
│   └── README.md
├── spotify_query.csv
└── tree.text
```
 
### Run a python file as a module vs script:
We need __init__.py in each sub-directory to have python recognize them as a package.
There are different ways to run a python file in the project. After activating the virtual env:
* Web app: `uvicorn appdir.main:app`
  - runs main as the top module. Sets main.__package__ = appdir,  and main.__name__ = __main__ . It also adds the full path of main.py containing directory, i.e. FastAPI-Spotify to sys.path.
* Run a module as a part of the top package: `FastAPI-Spotify % python -m appdir.tests.test_predict`
  - running as a module with the full path to the top package directory, sets test_predict.__package__ = appdir.tests,  and test_predict.__name__ = __main__. It also adds the full path of test_predict containing directory, i.e. “tests” to sys.path. However, it does not add the higher level directories to the path, and for ex/ appdir is not recognized.
* Run a .py file as a stand alone module: tests % python -m test_predict
  - sets test_predict.__package__ = tests,  and test_predict.__name__ = __main__. It also adds the full path of test_predict containing directory, i.e. “tests” to sys.path.
* Running a .py file as script: `FastAPI-Spotify % python appdir/tests/test_predict.py`
  - sets test_predict.__package__ = None and test_predict.__name__ = __main__. It also adds the full path of test_predict containing directory, i.e. “tests” to sys.path., but does not add the higher level directories to the sys.path.
* Running a .py file in vscode is similar to run it as a script on the terminal.
* Importing a .py file in python repl interactive session: 
``` 
<FastAPI-Spotify> python
>>> from appdir.tests import test_predict
```
  - test_predict.__package__ = “” (empty string) and the repl session takes the name “main”. sys.path would not contain any path to the project.

### Absolute path vs. Relative path::
* Relative path for a module in a package:
Let’s say inside test_predict.py we want to import a sibling module such as predict.py. If test_predict.__package__ is set to appdir.tests, the full name of the python file would be appdir.tests.test_predict. This would allow using a relative path as: `import ..api.predict` or `from ..api import predict`. The module's full name must have at least as many dots as there are in the import statement.
* Absolute path for a module in a package:
For importing predict.py from test_predict.py, we need to add <FastAPI-Spotify_path> to sys.path. Subdirectories are searchable due to __init__.py files and hence __package__ variables. `import appdir.api.predict` or `from appdir.api import predict`.
* Absolute path for a script:
Adding project directory to the path: `sys.path += <FastAPI-Spotify_path>`. 
* Absolute path for interactive session:
Here we don’t have the current directory path in the sys.path. However adding <FastAPI-Spotify_path> to sys.path would be sufficient as the __init__.py files construct the subdirectory structure
```
sys.path += [<FastAPI-Spotify_path>]
```

Adding the following code to the beginning of the python files that needs to import other modules, updates the system path with the project directory and also updates the __package__ variable which would allow all the above scenarios to run successfully.
```
if __name__ == '__main__' and  __package__ is None:
    print("is running as a Python Script")
else:
    print("is running as a Python Module")
if __package__ in [None, ""]:
    import re
    c_dir = re.sub(r"(^.*)\/.*\.py$", r"\g<1>", __file__)    
    from sys import path
    from os.path import dirname as dir
    path.append(dir(dir(c_dir)))
    __package__ = "appdir.api"
```

