# Transfer to Python

### Free Calculator

Now we will get our hands dirty to play with Python in the Command-Line. Python can be run in REPL(Read–eval–print loop). It is a simple way to code in Python. Type `python` without any parameters in the shell.

```bash
python
```

You will get something like this:

```python
Python 3.9.7 | packaged by conda-forge | (default, Sep 29 2021, 19:24:02)
[Clang 11.1.0 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

Try to type a simple calculation formula here.

```bash
>>> 26374 + 88140
114514
```

Then you have a simple CLI calculator! To use it as a calculator, `_` refers to the last output, just like the `ANS` in calculators.

```bash
>>> _ + 1805296
1919810
```

We can also set a variable and let it remember.

```bash
>>> a = -1
```

And we can just type `a` and we can get its value

```bash
>>> a
-1
```

### Writing Python Hello World.

Now using `Ctrl + D` to quit REPL. And let's try to write some Python code by Vim. First, `cd` to anywhere you like. Then:

```bash
vim hello.py
```

And press `i` to Enter INSERT mode to type the code.

```python
print("Hello, World")
```

Python allows you not to type `;` at the end of a line. Press the Esc key and type `:x` to exit Vim. Then run the code by:

```bash
python hello.py
```

### Complex Data Types

In Python, we have **List, Set, Dictionary, Tuple**.

```
>>> a = {1, 2, 3}
>>> type(a)
<class 'set'>

>>> b = [1, 2, 3]
>>> type(b)
<class 'list'>

>>> c = {1: 'a', 2: 'b'}
>>> type(c)
<class 'dict'>

>>> d = (1, 2, 3)
>>> type(d)
<class 'tuple'>
```

| **List**                                                                                                       | **Tuple**                                                                                            | **Set**                                                                             | **Dictionary**                                                                     |
| -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| List is a non-homogeneous data structure which stores the elements in single row and multiple rows and columns | Tuple is also a non-homogeneous data structure which stores single row and multiple rows and columns | Set data structure is also non-homogeneous data structure but stores in single row  | Dictionary is also a non-homogeneous data structure which stores key value pairs   |
| List can be represented by \[ ]                                                                                | <p>Tuple can be represented by  </p><p>( )</p>                                                       | Set can be represented by { }                                                       | Dictionary  can be represented by { }                                              |
| List allows duplicate elements                                                                                 | Tuple allows duplicate elements                                                                      | Set will not allow duplicate elements                                               | Set will not allow duplicate elements and dictionary doesn’t allow duplicate keys. |
| List can use nested among all                                                                                  | Tuple can use nested among all                                                                       | Set can use nested among all                                                        | Dictionary can use nested among all                                                |
| Example: \[1, 2, 3, 4, 5]                                                                                      | Example: (1, 2, 3, 4, 5)                                                                             | Example: {1, 2, 3, 4, 5}                                                            | Example: {1, 2, 3, 4, 5}                                                           |
| List can be created using **list()** function                                                                  | Tuple can be created using **tuple()** function.                                                     | Set can be created using **set()** function                                         | Dictionary can be created using **dict()** function.                               |
| List is mutable i.e we can make any changes in list.                                                           | Tuple  is immutable i.e we can not make any changes in tuple                                         | Set is mutable i.e we can make any changes in set. But elements are not duplicated. | Dictionary is mutable. But Keys are not duplicated.                                |
| List is ordered                                                                                                | Tuple is ordered                                                                                     | Set is unordered                                                                    | Dictionary is ordered                                                              |
| <p>Creating an empty list</p><p>l=[]</p>                                                                       | <p>Creating an empty Tuple</p><p>t=()</p>                                                            | <p>Creating a set</p><p>a=set()</p><p>b=set(a)</p>                                  | <p>Creating an empty dictionary</p><p>d={}</p>                                     |

To access a list item by index, the syntax is the same as Java.

```python
arr = [1, 2, 3, 4, 5]
print(arr[3])    # This will output 4
```

To work with dictionaries, we can use the key to get the value.

```python
dict = {"Alice": 12, "Bob": 13, "Charlie": 14}
print(dict["Alice"])    # This will output 12
```

### Loop and Condition

We want to print out all of the items in a list, we can simply use a for-each loop:

```python
arr = [1, 2, 3, 4, 5]
for a in arr:
    print(a)
```

Or something similar as Java:

```python
arr = [1, 2, 3, 4, 5]
for i in range(len(arr)):
    print(arr[i])
```

```java
int[] arr = {1, 2, 3, 4, 5};
for (int i = 0; i < arr.length; i++) {
    System.out.println(arr[i]);
}
```

Or If we want to loop a dictionary:

```python
dict = {"Alice": 12, "Bob": 13, "Charlie": 14}
for key in dict:
    print(key, dict[key])
```

