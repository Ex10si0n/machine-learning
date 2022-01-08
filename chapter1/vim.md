# Vim

**Vim** is a highly configurable text editor built to make creating and changing any kind of text very efficient. Let's have a demonstration of it.

{% embed url="https://www.youtube.com/watch?v=y6VJBeZEDZU" %}

#### Getting Start to use vim

Simply type

```bash
-> vim
```

And you can see an interface like this, this is the UI when you start `vim` by not specifying a file name.

```bash

~
~
~
~
~
~
~
~                                                       VIM - Vi IMproved
~
~                                                        version 8.1.2269
~                                                    by Bram Moolenaar et al.
~                                            Modified by team+vim@tracker.debian.org
~                                          Vim is open source and freely distributable
~
~                                                 Help poor children in Uganda!
~                                         type  :help iccf<Enter>       for information
~
~                                         type  :q<Enter>               to exit
~                                         type  :help<Enter>  or  <F1>  for on-line help
~                                         type  :help version8<Enter>   for version info
~
~
~
~
~
~
~
~
                                                                                                                0,0-1         All
```

Now type `:q` to quit Vim, you will find that when you type `:` your cursor will be redirected to the bottom of the window and then you can type **commands** in the input line. Then you will be back to the Linux shell.

Vim is a highly **hackable** and **customizable** text editor, so you can make your own configuration to this. Here is my Vim, I try to make it concise.

![](<../.gitbook/assets/Screenshot 2022-01-08 at 20.30.35.png>)

Now let us play around with Vim by editing a file. You should first `cd` to a directory you like to save the file that you will edit with Vim. Then type:

```bash
-> vim helloworld.py
```

This command will lead you to use Vim to create a file named `helloworld.py` and edit it at the same time. You are now able to edit the Python code (or script) with Vim. But take your time, you are now trying to type something but you find no response (or get some wired responses).

Now press `i` in the keyboard. Then you will find there is something like that in the bottom-left corner of the window. This means you are in **`INSERT` mode**, which means you can insert (type) anything in the buffer (file). Which looks like that. (You may not find the red bar but it is Okay when you notice that some text like `-- INSERT --` )

![](<../.gitbook/assets/Screenshot 2022-01-08 at 20.42.55 (1).png>)

Then type a simple hello-world snippet in the file you are editing, if you are not familiar with Python, please refer to the following code.

```python
print("Hello, World")
```

Since you finish editing the script, simply press  `Esc` key on the keyboard which is always located at the upper-left corner of the keyboard. You will notice that the `-- INSERT --` disappears. Now you are in the **`NORMAL` mode**. In this mode, you can navigate the cursor around the text, change a specific character, increase a number and play some magic.

Notice that almost all keys in the keyboard are mapped to a command or shortcut when you are in  **`NORMAL` mode**. What's more, you can even define more keys or keys-combination to do some other functions, like that: (which means when I press `control` + `t` , Vim will open a new Terminal window split in the downside of the current window)

```
nnoremap <C-t> :sp<CR><C-w><C-j> :term<CR>
```

I am used to mapping `jk` to `Esc` , which means when I quickly type `j` and then `k` , Vim will press for me `Esc` to back to the normal mode.

Vim has had a horrible learning curve since you started to learn it. It will take nearly **2-3 weeks** using it typing codes and texts to make it as your first editor. After that, you can configure it to make it adapt to you. Once you have managed it, you will find it is so handy.

![Leaning curves meme](../.gitbook/assets/7Cu9Z.jpg)

By the way, you can install Vim plugins into your favorite IDE, which means it can be used not only in the command line. Since Vim has been developed for over 40 years, the command and shortcuts are not changed, which means once you learn it, you can use it in your lifetime.

#### Vim Tutor

One of the easiest ways to learn Vim is by reading the official documentation. Typing

```bash
-> vimtutor
```

Try to play it around, try to switch your code (text) editor to Vim in the next few weeks.

> 👍 Goal: Finish `vimtutor` Lesson 1-4
>
> This part will be demostrated on the class.

#### Vim Cheat Sheets

![Basic Editing](../.gitbook/assets/vi-vim-tutorial-1.svg)

![Operations & Repeatitions](../.gitbook/assets/vi-vim-tutorial-2.svg)

![Yank(Copy) & Paste](<../.gitbook/assets/vi-vim-tutorial-3 (1).svg>)

![Searching](../.gitbook/assets/vi-vim-tutorial-4.svg)

![Marks & Macros](../.gitbook/assets/vi-vim-tutorial-5.svg)

![Misc](<../.gitbook/assets/vi-vim-tutorial-6 (1).svg>)

![Commands](../.gitbook/assets/vi-vim-tutorial-7.svg)

{% embed url="http://www.viemu.com/vi-vim-tutorial-svg.zip" %}
Download All
{% endembed %}

