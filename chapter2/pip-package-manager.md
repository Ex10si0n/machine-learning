# pip: Package manager

`pip` is the [package installer](https://packaging.python.org/guides/tool-recommendations/) for Python. You can use pip to install packages from the [Python Package Index](https://pypi.org) and other indexes.

The packages `pip` installed could be a Python module or executable binaries.

### Playaround

Let's have a look at `wikipedia` module, installation:

```bash
pip install wikipedia
```

Then we create a Python script with:

```python
import wikipedia
result = wikipedia.page("Macao Polytechnic Institute")
print(result.summary)
```

We can get:

```bash
Macao Polytechnic Institute (IPM; Chinese: 澳門理工學院; Portuguese: Instituto Politécnico de Macau) was established in 1981. It is located in the Macao Special Administrative Region of the People's Republic of China. MPI is a public Higher education institution with an emphasis on applied knowledge and skills.
```
