# Linux常见命令

##目录

+ cd / #进入根目录

  cd 或者cd home/进入主目录
  cd /etc/ 进入配置文件目录

  cd ..#直接进入根目录
  cd ~#进入用户主目录

+ mkdir 创建文件夹

+ rmdir 只能删除空文件夹

  rm 删除文件

+ pwd 显示当前目录

+ ls #列出文件和目录

  ls -l 列出文件的时候展示权限等信息

  ls -F#会在目录后面加/,可执行文件后面加*，在链接文件后面加@
  ls -a #用于显示隐含文件

## 文件

###创建及删除文件等

+ touch 创建文件#可以并列创建多个为你文件 touch file1 file2 file3

+ cp 拷贝文件# cp -i 会询问是否覆盖 cp -i file1 file2 or cp -i folder/file

  cp -R folder/ 拷贝整个文件夹的内容

  cp file1 file2 floder/#最后一个参数表示要拷贝到的地方

+ mv 剪切文件 

+ rm 删除文件

  **参数 -i 会询问**

+ rm -r floder 会删除文件夹及文件

### 编辑文件等

+ nano

  编辑文件

+ **vim**

  | 命令                                       | 功能说明                                                     |
  | ------------------------------------------ | ------------------------------------------------------------ |
  | 插入字符、行，执行下面操作后，进入编辑状态 |
  | a                                          | 进入插入模式，在光标所在处后面添加文本                       |
  | i                                          | 进入插入模式，在光标所在处前面添加文本                       |
  | A                                          | 进入插入模式，在光标所在行末尾添加文本                       |
  | I                                          | 进入插入模式，在光标所在行行首添加文本（非空字符前）         |
  | o                                          | 进入插入模式，在光标所在行下新建一行                         |
  | O                                          | 进入插入模式，在光标所在行上新建一行                         |
  | R                                          | 进入替换模式，覆盖光标所在处文本                             |
  | 剪切、粘贴、恢复操作                       |                                                              |
  | dd                                         | 剪切光标所在行                                               |
  | Ndd                                        | N代表一个数字，剪切从光标所在行开始的连续N行                 |
  | yy                                         | 拷贝光标所在行                                               |
  | Nyy                                        | N代表一个数字，复制从光标所在行开始的连续N行                 |
  | yw                                         | 复制从光标开始到行末的字符                                   |
  | Nyw                                        | N代表一个数字，复制从光标开始到行末的N个单词                 |
  | y^                                         | 复制从光标开始到行首的字符                                   |
  | y$                                         | 复制从光标开始到行末的字符                                   |
  | p                                          | 粘贴剪切板的内容在光标后（或所在行的下一行，针对整行复制）   |
  | P                                          | 粘贴剪切板的内容在光标前（或所在行的上一行，针对整行复制）   |
  | u                                          | 撤销上一步所做的操作                                         |
  | 保存、退出、打开多个文件                   |                                                              |
  | :q!                                        | 强制退出，不保存                                             |
  | :w                                         | 保存文件，使用:w file，将当前文件保存为file                  |
  | :wq                                        | 保存退出                                                     |
  | :new                                       | 在当前窗口新建一个文本，使用:new file，打开file文件，使用Ctrl+ww在多个窗口间切换 |
  | 设置行号，跳转                             |                                                              |
  | :set nu                                    | 显示行号，使用:set nu!或:set nonu可以取消显示行号            |
  | n+                                         | 向下跳n行                                                    |
  | n-                                         | 向上跳n行                                                    |
  | nG                                         | 跳到行号为n的行                                              |
  | G                                          | 跳到最后一行                                                 |
  | H                                          | 跳到第一行                                                   |
  | 查找、替换                                 |                                                              |
  | /***                                       | 查找并高亮显示***的字符串，如/abc                            |
  | :s                                         | :s/old/new//,用new替换行中首次出现的old:s/old/new/g,用new替换行中所有的old:n,m s/old/new/g,用new替换从n到m行中所有new:%s/old/new/g,用new替换当前文件中所有old |

+ cat

  cat file 显示文件内容

  cat file1 > file2 #把文件一的内容复制放到文件2里面

  cat file1 file2 >file3 #把文件一二打包放到文件3里面

  cat file1>>file2 #把文件一追加到文件2后面

### 文件权限

+ ls -l #展示所有权限

  前三位user权限，中三位group权限，后三位other权限

+ -x 表示可执行 -w 表示可写 -r 表示可读

+ 执行命令: ./

+ chmod u+x file1 #即将user的可执行权限给file1

  u-x 表示去掉权限

  a-r 表示所有文件去掉可读权限

  ug +x 表示user和group都加上可执行权限

  **python文件第一行加入#!/usr/bin/python即可以执行**

###解压缩相关

+ tar类型

  ```python
  tar -cf all.tar *.txt 将所有txt文件打包成all.-c是表示产生新的包 ，-f指定包的文件名
  
  tar -rf all.tar *.txt 将所有.txt的文件增加到all.tar,-r表示新增
  
  tar -tf all.tar 列出all.tar包中所有文件，-t表示列出
  
  tar -xzf all.tar.gz 解压该包
  ```

+ 对于不同的压缩文件

  + 对于.tar结尾的文件

    tar -xf all.tar

  + 对于.gz结尾的文件

    gzip -d all.gz

    gunzip all.gz

  + 对于.tgz或.tar.gz结尾的文件

    tar -xzf all.tar.gz

    tar -xzf all.tgz

  + 对于.bz2结尾的文件

    bzip2 -d all.bz2

    bunzip2 all.bz2

  + 对于tar.bz2结尾的文件

    tar -xjf all.tar.bz2

  + 对于.Z结尾的文件

    uncompress all.Z

  + 对于.tar.Z结尾的文件

    tar -xZf all.tar.z

+ 对于tar不同参数的补充

  ```python
  独立参数
  -c: 建立压缩档案 
  -x：解压 
  -t：查看内容 
  -r：向压缩归档文件末尾追加文件 
  -u：更新原压缩包中的文件
  #这五个是独立的命令，压缩解压都要用到其中一个，可以和别的命令连用但只能用其中一个
  可选参数
  -z：有gzip属性的 
  -j：有bz2属性的 
  -Z：有compress属性的 
  -v：显示所有过程 
  -O：将文件解开到标准输出 
  必选参数
  -f：表示文件
  ```

+ <font color='red'>对于windows常见的压缩包，例如.zip和.rar格式的，如下</font>

  + .zip

    ```
    zip file1.zip *.txt 同上，压缩
    unzip file1.zip 解压缩
    ```

  + .rar <font color='red'>需先下载linux for windows</font>

    ```python
    rar file1.rar *.txt 同上，压缩
    unrar file1.rar 解压缩
    ```

