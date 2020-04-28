from shutil import copyfile
import os



file = '/home/paulin/Documents/datas/celeb/list_attr_celeba.txt' 

Lsmile = []
Lnotsmile = []

nb = 60000

f= open(file,"r")
line = f.readline()
a = line
i = 0
while line :
    line = f.readline()
    line_list = line.split(" ")

    #print(line_list)
    line_list = list(filter(lambda a: a != '', line_list))
    #print(line_list)

    if i == 0 :
        c = line_list
    if i == 1 :
        b = line_list
    if len(line_list) < 30:
        print('stop a'+str(i))
        break
    if len(Lsmile) < nb :
        if line_list[32] == '1' :
            Lsmile.append(line_list[0])
    if len(Lnotsmile) < nb :
        if line_list[32] == '-1' :
            Lnotsmile.append(line_list[0])
    i += 1
f.close()

print(c)
print(b)
print(len(c))
print(len(b))


print(len(Lsmile))
print(len(Lnotsmile))

for x in Lsmile :
    if x in Lnotsmile :
        print('pas ok')

Limg = os.listdir("/home/paulin/Documents/datas/celeb/celeb_168_216")
files = [str(i).zfill(6)+".jpg" for i in range (1, 200000)]
print(len(Limg))
#print(Limg)
i = 0
for filename in Limg:
    i+= 1
    if i % 1000 == 0:
        print(i)
    if i < 1000000 :
        if filename in Lsmile:
            copyfile("/home/paulin/Documents/datas/celeb/celeb_168_216/"+filename, '/home/paulin/Documents/datas/celeb/celeb_smile_168_216/celebSmile/'+filename)
        if filename in Lnotsmile: 
            copyfile("/home/paulin/Documents/datas/celeb/celeb_168_216/"+filename, '/home/paulin/Documents/datas/celeb/celeb_smile_168_216/celebNotSmile/'+filename)

print('ok')