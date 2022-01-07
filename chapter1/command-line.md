# Command Line

Let have the first try on using Linux. Although Linux has a GUI(Graphical User Interface) as Windows and macOS, the server always disable it due to the heavy resource load by GUI. So you have to ensure yourself familiar with the Command Line Operation.

> Tips: Although you can use my Linux Server now, I strongly advise you try to install it in your own  Laptop. Since you can enjoy the configuration of your own Computer System.

For students at Macao Polytechnic Institute Interest Group, you are welcome to use my server before you have installed one on your computer. Here is a tip on how to open the CLI(Command Line Interface).

#### macOS

Using Spotlight Search **(⌘ + Space)**. Open `Terminal.app` and you can see an interactive Interface with a rectangle cursor.

![](<../.gitbook/assets/Screenshot 2022-01-07 at 18.16.12.png>)

#### Windows

Right-click Start Button, open `Windows Terminal` or `Windows Powershell` or `CMD` . 如果你在使用 Windows 中文版，则是`Windows Powershell` 或者 `命令提示符`.

![](../.gitbook/assets/win11\_launch\_windows\_terminal.jpg.webp)

### Get Started with CLI

Since you have already opened your Terminal Software, try to type `ssh` in the interactive line. And you will get something like this.

```bash
$ ssh
usage: ssh [-46AaCfGgKkMNnqsTtVvXxYy] [-B bind_interface]
           [-b bind_address] [-c cipher_spec] [-D [bind_address:]port]
           [-E log_file] [-e escape_char] [-F configfile] [-I pkcs11]
           [-i identity_file] [-J [user@]host[:port]] [-L address]
           [-l login_name] [-m mac_spec] [-O ctl_cmd] [-o option] [-p port]
           [-Q query_option] [-R address] [-S ctl_path] [-W host:port]
           [-w local_tun[:remote_tun]] destination [command]
```

After having a brief read of the instruction, you should be clear that if the command `ssh` is installed on your computer (If you are using Windows 10 or 11, any versions of macOS, you should have this command by the OS)

Then we shall use it to log in to my Linux Server.

```bash
ssh ubuntu@150.158.151.180
```

And you will get this, just type the password I gave you, but you will not see what characters you have typed due to the security.

```
ubuntu@150.158.151.180's password: []
```

Then you will get this output:

```bash
Welcome to Ubuntu 20.04.3 LTS (GNU/Linux 5.4.0-92-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

  System information as of WW MM  hh:mm:ss YYYY PM CST

  System load:  0.07               Processes:                138
  Usage of /:   18.1% of 78.69GB   Users logged in:          0
  Memory usage: 19%                IPv4 address for docker0: 172.17.0.1
  Swap usage:   0%                 IPv4 address for eth0:    10.0.12.13


Last login: WW MM  hh:mm:ss YYYY from XXX.XXX.XXX.XXX
Welcome to fish, the friendly interactive shell
Type `help` for instructions on how to use fish

ubuntu on VM-12-13-ubuntu at home via pythonv3.9.7
->
```

This is the welcome message by Ubuntu. And now, you are manipulating my Server rather than your computer via the Internet connection. You can play around with my Server freely without malicious intention. So do not do some stupid kinds of stuff like deleting my file system.

Now let's try to type some commands. What you type is after the -> Symbol. So in the following example, the command is only `ls` and what you can see from the shell after you type `ls` is `hello.txt` which is the output of the shell.

```bash
->  ls
hello.txt
```

> Google search: `shell vs terminal`

This `hello.txt` is a file that is located at the entry directory when you `ssh` into my Server. To see what it contains, the very simple way is `cat` (means to concatenate the file content in the output)

```bash
-> cat hello.txt
==============================================================================
Hello Everyone, Welcome to my Interest Group on `LINUX AND PYTHON AI TOOLKITS`
==============================================================================
You can see this file's content by typing `cat hello.txt` command.
commands
```

Image `->` as a mouse pointer, it is the command line prompt. Use `pwd` to see where you are.

```bash
-> pwd
/home/ubuntu
```

Now let us create your own folders with your name **(change** `steve-yan` **to your name)**

```bash
-> mkdir steve-yan
```

And go inside your folder

```bash
ubuntu on VM-12-13-ubuntu at home
-> cd steve-yan
```

Then you are at

```bash
ubuntu on VM-12-13-ubuntu at home/steve-yan
-> pwd
/home/ubuntu/steve-yan
```

Actually, we can see it before the `->` prompt `at home/steve-yan` , `home` refers to the `ubuntu` directory. You can think of it as Windows' `C:\Users\Ex10si0n` or macOS's `/Users/ex10si0n` .

#### Linux File Structure

![Linux file system](<../.gitbook/assets/linux-file-system-hierarchy-1 (1).png>)

