# Importing modules

One feature of Python that makes it useful for a wide range of tasks is the fact that it comes "batteries included" – that is, the Python standard library contains useful tools for a wide range of tasks. On top of this, there is a broad ecosystem of third-party tools and packages that offer more specialized functionality. Here we'll take a look at importing standard library modules, tools for installing third-party modules, and a description of how you can make your own modules.&#x20;

For loading built-in and third-party modules, Python provides the `import` statement. There are a few ways to use the statement, which we will briefly mention in this chapter, from most recommended to least recommended.

### Explicit module import&#x20;

Explicit import of a module preserves the module's content in a namespace. The namespace is then used to refer to its contents with a "`.`" between them. For example, here we'll import the built-in `math` module and compute the cosine of pi:&#x20;

```python
import math
math.cos(math.pi) 
```

### Explicit module import by the alias&#x20;

For longer module names, it's not convenient to use the full module name each time you access some element. For this reason, we'll commonly use the "`import ... as ...`" pattern to create a shorter alias for the namespace. For example, the NumPy (Numerical Python) package, a popular third-party package useful for data science, is by convention imported under the alias np:&#x20;

```python
import numpy as np
np.cos(np.pi) 
```

### Explicit import of module contents

Sometimes rather than importing the module namespace, you would just like to import a few particular items from the module. This can be done with the "`from ... import ...`" pattern. For example, we can import just the cos function and the pi constant from the math module:&#x20;

```python
from math import cos, pi
cos(pi)
```

### Implicit import of module contents

Finally, it is sometimes useful to import the entirety of the module contents into the local namespace. This can be done with the "`from ... import *`" pattern:&#x20;

```python
from math import * 
sin(pi) ** 2 + cos(pi) ** 2 
```

This pattern should be used sparingly, if at all. The problem is that such imports can sometimes overwrite function names that you do not intend to overwrite, and the implicitness of the statement makes it difficult to determine what has changed.

### Importing from the Python's standard library

Python's standard library contains many useful built-in modules, which you can read about fully in _Python's documentation_. Any of these can be imported with the `import` statement, and then explored using the help function seen on the previous page. Here is an extremely incomplete list of some of the modules you might wish to explore and learn about:

* `os` and `sys`: Tools for interfacing with the operating system, including navigating file directory structures and executing shell commands&#x20;
* `math` and `cmath`: Mathematical functions and operations on real and complex numbers&#x20;
* `itertools`: Tools for constructing and interacting with iterators and generators
* `functools`: Tools that assist with functional programming&#x20;
* `random`: Tools for generating pseudorandom numbers&#x20;
* `pickle`: Tools for object&#x20;
* `persistence`: saving objects to and loading objects from the disk
* `json` and `csv`: Tools for reading JSON-formatted and CSV-formatted files.&#x20;
* `urllib`: Tools for doing HTTP and other web requests.

### Importing from third-party modules

Installation in CLI.

```bash
pip install numpy
```

Import in code.

```python
import numpy as np
```

