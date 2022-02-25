# DS Utilities: Numpy

## TD; DR

![Numpy](https://numpy.org/images/logo.svg)

In this Chapter, take the following 25 exercises to learn how to use the basic APIs of `numpy`.

***
## Exercise on Numpy

#### 1. Import the numpy package under the name `np`


```python
import numpy as np
```

#### 2. Create a null vector of size 10


```python
Z = np.zeros(10)
print(Z)
```

#### 3. Create a null vector of size 10 but the fifth value which is 1 


```python
Z = np.zeros(10)
Z[4] = 1
print(Z)
```

#### 4. Create a vector with values ranging from 10 to 49


```python
Z = np.arange(10,50)
print(Z)
```

#### 5. Reverse a vector (first element becomes last)


```python
Z = np.arange(10)
Z = Z[::-1]
print(Z)
```

#### Also tries:

```python
Z = np.arange(10)
Z = Z[::-2]
print(Z)
```

```python
Z = np.arange(10)
Z = Z[::3]
print(Z)
```

#### 6. Create a 3x3 matrix with values ranging from 0 to 8

```python
Z = np.arange(9).reshape(3, 3)
print(Z)
```

#### 7. Find indices of non-zero elements from `[1,2,0,0,4,0]`


```python
nz = np.nonzero([1,2,0,0,4,0])
print(nz)
```

#### 8. Create a 3x3 identity matrix

> Identity matrix: values on **positive diagnosis** of matrix are 1, and others are 0
>
> Also recognized as I_n


```python
Z = np.eye(3)
print(Z)
```

#### 9. Create a 3x3x3 array with random values


```python
Z = np.random.random((3,3,3))
print(Z)
```

#### 10. Create a 10x10 array with random values and find the minimum and maximum values


```python
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)
```

#### 11. Create a random vector of size 30 and find the mean value 


```python
Z = np.random.random(30)
m = Z.mean()
print(m)
```

#### 12. Create a 2d array with 1 on the border and 0 inside 


```python
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)
```

#### 13. How to add a border (filled with 0's) around an existing array?

```python
Z = np.arange(25).reshape(5, 5)
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
print(Z)
```

#### 14. Create a 5x5 matrix with values 0,1,2,3,4 on the diagonal

```python
Z = np.diag(np.arange(5))
print(Z)
```

#### 15. Create a 8x8 matrix and fill it with a chess board pattern

> Chess board Pattern is just like:
>
> ```
> 0 1 0 1 0 1 0 1
> 1 0 1 0 1 0 1 0
> 0 1 0 1 0 1 0 1
> 1 0 1 0 1 0 1 0
> 0 1 0 1 0 1 0 1
> 1 0 1 0 1 0 1 0
> 0 1 0 1 0 1 0 1
> 1 0 1 0 1 0 1 0
> ```

```python
Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)

# Alternative

Z = np.tile(np.array([[0,1],[1,0]]), (4,4))
print(Z)
```

#### 16. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?


```python
print(np.unravel_index(99,(6,7,8)))
```

#### 17. Matrix Addition and Subtraction

Add up two matries A and B, where A and B respectively be:

```
A = [[1 2 3]
     [4 5 6]
     [7 8 9]]
     
B = [[9 8 7]
     [6 5 4]
     [3 2 1]]
```

```python
A = np.arange(1, 10).reshape(3, 3)
B = np.arange(9, 0, -1).reshape(3, 3)
print(A + B)
```

#### 18. Normalize a 5x5 random matrix

> **Normalize a matrix**(归一化矩阵) is to make every values in the matrix lies on [min, max], as an example: we have an array valued [-1, 1, 3]. After normalization, it becomes [0, 0.5, 1].


```python
Z = np.random.random((5, 5))
Z = (Z - Z.min())/(Z.max() - Z.min())
print(Z)
```

#### 19. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)

```python
Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)
```

#### 20. Given a 1D array, negate all elements which are between 3 and 8, in place.


```python
Z = np.arange(11)
Z[(3 < Z) & (Z < 8)] *= -1
print(Z)
```

#### 21. How to find common values between two arrays? 

```python
Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(Z1, Z2)
print(np.intersect1d(Z1,Z2))
```

#### 22. How to get the dates of yesterday, today and tomorrow? 

```python
yesterday = np.datetime64('today') - np.timedelta64(1)
today     = np.datetime64('today')
tomorrow  = np.datetime64('today') + np.timedelta64(1)
print(yesterday, today, tomorrow)
```

#### 23. Create a vector of size 6 with values in equal spacing ranges [0, 20]

```python
Z = np.linspace(0,20,6)
print(Z)
```

#### 24. Create a random vector of size 10 and sort it


```python
Z = np.random.random(10)
Z.sort()
print(Z)
```

#### 25. Create random vector of size 10 and replace the maximum value by 0


```python
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)
```

> What is the relationship between `Z.argmax()` and `Z.max()`?
>
> ```python
> Z = np.random.random(10)
> print(Z[Z.argmax()] == Z.max())
> ```

