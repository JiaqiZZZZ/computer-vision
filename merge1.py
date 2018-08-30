from cartoon import *
from functions import *


'''
Program usage:  "python merge1.py demo" or "python merge1.py v1={x} v2={y} v3={z} v4={w}"
(v1, v2, v3 can be in any order, no space permitted between '=' and varables)
x range from 00-17
y range from 0-9
z only has 00
w only values '19'


There are 2 mode in command line, one is demo which will demo all the things we have done 
individually without combined status. If u want combined states and take a photo, use second
mode.

v1 is 17 stickers
v2 is 10 filters
v3 is photo frame
v4 is dynamic sticker

Note that v2 could not combine with other mode.
Distortion mirror funtion is only in demo mode, which is the last camera mode.
'''

if __name__ =='__main__':
    if len(sys.argv)>1 and sys.argv[1] == 'demo':
        v1_values = ['0'+str(i) for i in range(9)]
        v1_values = v1_values + [str(i) for i in range(10,18)]
        v2_values = [str(i) for i in range(10)]
        v3_values = ['00']
        mode_dic = {'v1': None, 'v2': None, 'v3': None, 'v4': None,'smallerface': False}

        image_path = 'train.jpg'
        print("only v1")
        for i in v1_values:
            mode_dic['v1'] = i 
            demo(mode_dic)
            # img = run_image(mode_dic,image_path)
            # cv2.imwrite('report_image/v1_'+i+'.jpg', img)

        mode_dic['v1'] = None
        print("only v2")
        for i in v2_values:
            mode_dic['v2'] = i
            demo(mode_dic)
            # img = run_image(mode_dic,image_path)
            # cv2.imwrite('report_image/v2_'+str(i)+'.jpg', img)
            # print(i)
        mode_dic['v2'] = None
        print("only v3")
        for i in v3_values:
            mode_dic['v3'] = i 

            # mode_dic['v1'] = '03'
            demo(mode_dic)
        mode_dic['v3'] = None
        print("only v4")
        dynamicSticker('19')
        print("HAHA")
        mode_dic['smallerface'] = True
        run_camera(mode_dic)

    else:
        '''
        v1 = '00' - '19'
        v2 = '0' - '9'
        v3 = '00'
        v4 = '19'
        '''
        mode_dic = {'v1': None, 'v2': None, 'v3': None, 'v4': None, 'smallerface': False}
        for i in range(1,len(sys.argv)):
            mode_pair = sys.argv[i].replace(' ','')
            mode_pair = mode_pair.split('=')
            if mode_pair[0] not in mode_dic:
                print("Command Line Error!")
                print("Usage: merge1.py v1= v2= v3= v4=")
            mode_dic[mode_pair[0]] = mode_pair[1]

        if mode_dic['v4'] != None:
            dynamicSticker(mode_dic['v4'])

        else:
            # run_image(mode_dic)
            run_camera(mode_dic)
            # dynamicSticker('19')



