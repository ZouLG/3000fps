# 3000fps
A Visual Studio version of re-implementation of the paper "Face alignment at 3000fps...".

Usage: Modify the path of training dataset. The folder of the dataset should include a text file named Path_Images.txt, which records all the image file path of the dataset, e.g.

D:\Projects_Face_Detection\Datasets\helen\trainset\100032540_1.jpg D:\Projects_Face_Detection\Datasets\helen\trainset\100040721_1.jpg D:\Projects_Face_Detection\Datasets\helen\trainset\100040721_2.jpg D:\Projects_Face_Detection\Datasets\helen\trainset\1002681492_1.jpg D:\Projects_Face_Detection\Datasets\helen\trainset\1004467229_1.jpg D:\Projects_Face_Detection\Datasets\helen\trainset\100466187_1.jpg D:\Projects_Face_Detection\Datasets\helen\trainset\100591971_1.jpg D:\Projects_Face_Detection\Datasets\helen\trainset\100843687_1.jpg D:\Projects_Face_Detection\Datasets\helen\trainset\1010057391_1.jpg D:\Projects_Face_Detection\Datasets\helen\trainset\1012675629_1.jpg D:\Projects_Face_Detection\Datasets\helen\trainset\1012675629_2.jpg D:\Projects_Face_Detection\Datasets\helen\trainset\1018882799_1.jpg

You can adjust your cmd directory to your dataset and enter: dir /b/s/p/w *.jpg>Path_Images.txt to get the text file.

After that, just run it.
