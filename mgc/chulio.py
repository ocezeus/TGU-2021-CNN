import os
From_file=open('keywords.csv')
f=open('niuniu1.txt','w')
count=0
huancun=[]
for each_line in From_file:
 #print(type(each_line)) each_line 是字符类型
 Delstr=list(each_line)
 Delstr.pop() #弹出n
 Delstr.pop() #弹出\
 Delstr.append(',\n')
 huancun="".join(Delstr)
 print(huancun)
 f.writelines(huancun)
 count+=1
 huancun=[]
f.close()
From_file.close()
print('文件中总共有：%d行'%count)