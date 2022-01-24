# conda: the Python environment manager

### **Installing conda:**

There are several ways to make `conda` runs on your computer. Here are two tastes, **Anaconda** is a data science toolkit all-in-one edition that makes sure you are ready to go once you have installed it. While **MiniForge** needs more configurations but it is pure and minimal, the package size is smaller, compared with Anaconda, MiniForge will only install `conda` for you.

> Anaconda (coming with essential DS tools): [https://www.anaconda.com](https://www.anaconda.com)
>
> MiniForge (minimal edition): [https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge)

Alternatively, you can use the command line to install conda:

**macOS:**

Needs to install macOS Package Manager `brew` from [https://brew.sh](https://brew.sh)

```bash
# Install Homebrew (It takes long time)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Then
brew install miniforge
```

**Windows:**

Need to install a package manager `choco` from [https://chocolatey.org](https://chocolatey.org)

```powershell
# Install choco Package Manager first
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Then (in Admin Powershell)
choco install anaconda3
```

The Linux machine I have provided is already installed `conda` binary. If you are now using Linux in lanqiao.cn, the experiment environment is not connected to the Internet so I recommend you install it on your own PC.

Once you have installed the conda, please make sure you can type `conda` in the CLI. If not,

```bash
# path/to/conda refers to conda bin where you have installed
path/to/conda init
```

### Environment

The environment is a collection of Python and its utilities. It enables you to use different versions of Python on one computer. The following command is for creating an environment named `pyml`

```bash
conda create -n pyml python=3.8
```

`-n` refers to specify a name for this environment and `python=3.8` refers to this environment needs Python version at 3.8

Then, we can change our current environment to `pyml` using

```bash
conda activate pyml
```

If you are always want to use this environment, you can add this line to your shell config file (`.zshrc` or `.bashrc` ), and when you open your shell, the environment is activated automatically.

Have a look at the `python` executive.

```bash
which python
```

You can have a path returned by the last command, and it will like:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/webapi/bin/python
```

That's it, conda can change the `python` executive path whenever you activate an environment.

If you want to delete the environment, use:

```bash
conda env remove -n <env_name>
```
