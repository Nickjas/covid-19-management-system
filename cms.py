import tkinter as tk
from tkinter import *
import cv2
import csv
import os
import numpy as np
from PIL import Image,ImageTk
import pandas as pd
import datetime
import time
from scipy.spatial import distance as dist
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from mtcnn.mtcnn import MTCNN
import matplotlib as plt
from imutils.video import VideoStream
import imutils

#####Window is our Main frame of system
window = tk.Tk()
window.title("CMS-Covid Management System")

window.geometry('1280x720')
window.configure(background='green')

####GUI for manually fill attendance

def manually_fill():
    global sb
    sb = tk.Tk()
    #sb.iconbitmap('AMS.ico')
    sb.title("Enter subject name...")
    sb.geometry('580x320')
    sb.configure(background='snow')

    def err_screen_for_subject():

        def ec_delete():
            ec.destroy()
        global ec
        ec = tk.Tk()
        ec.geometry('300x100')
        #ec.iconbitmap('AMS.ico')
        ec.title('Warning!!')
        ec.configure(background='snow')
        Label(ec, text='Please enter your subject name!!!', fg='red', bg='white', font=('times', 16, ' bold ')).pack()
        Button(ec, text='OK', command=ec_delete, fg="black", bg="lawn green", width=9, height=1, activebackground="Red",
               font=('times', 15, ' bold ')).place(x=90, y=50)

    def fill_attendance():
        ts = time.time()
        Date = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour, Minute, Second = timeStamp.split(":")
        ####Creatting csv of attendance

        ##Create table for Attendance
        date_for_DB = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d')
        global subb
        subb=SUB_ENTRY.get()
        #REGISTER= str(subb + "_" + Date + "_Time_" + Hour + "_" + Minute + "_" + Second)

        import psycopg2

        ###Connect to the database
        try:
            global cursor
            connection = psycopg2.connect(host="localhost",database="postgres",user="postgres",password="nickson1992")
            cursor = connection.cursor()
        except Exception as e:
            print(e)

        sql =  """CREATE TABLE IF NOT EXISTS REGISTER
                        (ID INT NOT NULL ,
                         ENROLLMENT varchar(100) NOT NULL,
                         NAME VARCHAR(50) NOT NULL,
                         DATE VARCHAR(20) NOT NULL,
                         TIME VARCHAR(20) NOT NULL,
                             PRIMARY KEY (ID)
                             );
                        """


        try:
            cursor.execute(sql)  ##for create a table
        except Exception as ex:
            print(ex)  #

        if subb=='':
            err_screen_for_subject()
        else:
            sb.destroy()
            MFW = tk.Tk()
            #MFW.iconbitmap('AMS.ico')
            MFW.title("Manually attendance of "+ str(subb))
            MFW.geometry('880x470')
            MFW.configure(background='snow')

            def del_errsc2():
                errsc2.destroy()

            def err_screen1():
                global errsc2
                errsc2 = tk.Tk()
                errsc2.geometry('330x100')
                #errsc2.iconbitmap('AMS.ico')
                errsc2.title('Warning!!')
                errsc2.configure(background='snow')
                Label(errsc2, text='Please enter Student & Enrollment!!!', fg='red', bg='white',
                      font=('times', 16, ' bold ')).pack()
                Button(errsc2, text='OK', command=del_errsc2, fg="black", bg="lawn green", width=9, height=1,
                       activebackground="Red", font=('times', 15, ' bold ')).place(x=90, y=50)

            def testVal(inStr, acttyp):
                if acttyp == '1':  # insert
                    if not inStr.isdigit():
                        return False
                return True

            ENR = tk.Label(MFW, text="Enter Enrollment", width=15, height=2, fg="white", bg="blue2",
                           font=('times', 15, ' bold '))
            ENR.place(x=30, y=100)

            STU_NAME = tk.Label(MFW, text="Enter Student name", width=15, height=2, fg="white", bg="blue2",
                                font=('times', 15, ' bold '))
            STU_NAME.place(x=30, y=200)

            global ENR_ENTRY
            ENR_ENTRY = tk.Entry(MFW, width=20,validate='key', bg="yellow", fg="red", font=('times', 23, ' bold '))
            ENR_ENTRY['validatecommand'] = (ENR_ENTRY.register(testVal), '%P', '%d')
            ENR_ENTRY.place(x=290, y=105)

            def remove_enr():
                ENR_ENTRY.delete(first=0, last=22)

            STUDENT_ENTRY = tk.Entry(MFW, width=20, bg="yellow", fg="red", font=('times', 23, ' bold '))
            STUDENT_ENTRY.place(x=290, y=205)

            def remove_student():
                STUDENT_ENTRY.delete(first=0, last=22)

            ####get important variable
            def enter_data_DB():
                ENROLLMENT = ENR_ENTRY.get()
                STUDENT = STUDENT_ENTRY.get()
                if ENROLLMENT=='':
                    err_screen1()
                elif STUDENT=='':
                    err_screen1()
                else:
                    time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    Hour, Minute, Second = time.split(":")
                    Insert_data = "INSERT INTO REGISTER (ID,ENROLLMENT,NAME,DATE,TIME) VALUES (0, %s, %s, %s,%s)"
                    VALUES = (str(ENROLLMENT), str(STUDENT), str(Date), str(time))
                    try:
                        cursor.execute(Insert_data, VALUES)
                    except Exception as e:
                        print(e)
                    ENR_ENTRY.delete(first=0, last=22)
                    STUDENT_ENTRY.delete(first=0, last=22)

            def create_csv():
                import csv
                cursor.execute("select * from  REGISTER ;")
                csv_name='C:/Users/HP/Desktop/covid/Attendace managemnt system/Attendance/Manually Attendance/ attendance.csv'
                with open(csv_name, "w") as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([i[0] for i in cursor.description])  # write headers
                    csv_writer.writerows(cursor)
                    O="CSV created Successfully"
                    Notifi.configure(text=O, bg="Green", fg="white", width=33, font=('times', 19, 'bold'))
                    Notifi.place(x=180, y=380)
                import csv
                import tkinter
                root = tkinter.Tk()
                root.title("Attendance of " + subb)
                root.configure(background='snow')
                with open(csv_name, newline="") as file:
                    reader = csv.reader(file)
                    r = 0

                    for col in reader:
                        c = 0
                        for row in col:
                            # i've added some styling
                            label = tkinter.Label(root, width=13, height=1, fg="black", font=('times', 13, ' bold '),
                                                  bg="lawn green", text=row, relief=tkinter.RIDGE)
                            label.grid(row=r, column=c)
                            c += 1
                        r += 1
                root.mainloop()

            Notifi = tk.Label(MFW, text="CSV created Successfully", bg="Green", fg="white", width=33,
                                height=2, font=('times', 19, 'bold'))


            c1ear_enroll = tk.Button(MFW, text="Clear", command=remove_enr, fg="black", bg="deep pink", width=10,
                                     height=1,
                                     activebackground="Red", font=('times', 15, ' bold '))
            c1ear_enroll.place(x=690, y=100)

            c1ear_student = tk.Button(MFW, text="Clear", command=remove_student, fg="black", bg="deep pink", width=10,
                                      height=1,
                                      activebackground="Red", font=('times', 15, ' bold '))
            c1ear_student.place(x=690, y=200)

            DATA_SUB = tk.Button(MFW, text="Enter Data",command=enter_data_DB, fg="black", bg="lime green", width=20,
                                 height=2,
                                 activebackground="Red", font=('times', 15, ' bold '))
            DATA_SUB.place(x=170, y=300)

            MAKE_CSV = tk.Button(MFW, text="Convert to CSV",command=create_csv, fg="black", bg="red", width=20,
                                 height=2,
                                 activebackground="Red", font=('times', 15, ' bold '))
            MAKE_CSV.place(x=570, y=300)

            def attf():
                import subprocess
                subprocess.Popen(r'explorer /select,"C:\Users\HP\Desktop\covid\Attendace managemnt system\Attendance\Manually Attendance\-------Check atttendance-------"')

            attf = tk.Button(MFW,  text="Check Sheets",command=attf,fg="black"  ,bg="lawn green"  ,width=12  ,height=1 ,activebackground = "Red" ,font=('times', 14, ' bold '))
            attf.place(x=730, y=410)

            MFW.mainloop()


    SUB = tk.Label(sb, text="Enter Subject", width=15, height=2, fg="white", bg="blue2", font=('times', 15, ' bold '))
    SUB.place(x=30, y=100)

    global SUB_ENTRY

    SUB_ENTRY = tk.Entry(sb, width=20, bg="yellow", fg="red", font=('times', 23, ' bold '))
    SUB_ENTRY.place(x=250, y=105)

    fill_manual_attendance = tk.Button(sb, text="Fill Attendance",command=fill_attendance, fg="white", bg="deep pink", width=20, height=2,
                       activebackground="Red", font=('times', 15, ' bold '))
    fill_manual_attendance.place(x=250, y=160)
    sb.mainloop()

##For clear textbox
def clear():
    txt.delete(first=0, last=22)

def clear1():
    txt2.delete(first=0, last=22)
def del_sc1():
    sc1.destroy()
def err_screen():
    global sc1
    sc1 = tk.Tk()
    sc1.geometry('300x100')
    sc1.iconbitmap('AMS.ico')
    sc1.title('Warning!!')
    sc1.configure(background='snow')
    Label(sc1,text='Enrollment & Name required!!!',fg='red',bg='white',font=('times', 16, ' bold ')).pack()
    Button(sc1,text='OK',command=del_sc1,fg="black"  ,bg="lawn green"  ,width=9  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold ')).place(x=90,y= 50)

##Error screen2
def del_sc2():
    sc2.destroy()
def err_screen1():
    global sc2
    sc2 = tk.Tk()
    sc2.geometry('300x100')
    sc2.iconbitmap('AMS.ico')
    sc2.title('Warning!!')
    sc2.configure(background='snow')
    Label(sc2,text='Please enter your subject name!!!',fg='red',bg='white',font=('times', 16, ' bold ')).pack()
    Button(sc2,text='OK',command=del_sc2,fg="black"  ,bg="lawn green"  ,width=9  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold ')).place(x=90,y= 50)

###For take images for datasets
def take_img():
    l1 = txt.get()
    l2 = txt2.get()
    if l1 == '':
        err_screen()
    elif l2 == '':
        err_screen()
    else:
        try:
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            Enrollment = txt.get()
            Name = txt2.get()
            sampleNum = 0
            while (True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # incrementing sample number
                    sampleNum = sampleNum + 1
                    # saving the captured face in the dataset folder
                    cv2.imwrite("TrainingImages/ " + Name + "." + Enrollment + '.' + str(sampleNum) + ".jpg",
                                gray[y:y + h, x:x + w])
                    cv2.imshow('Frame', img)
                # wait for 100 miliseconds
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # break if the sample number is morethan 100
                elif sampleNum > 10:
                    break
            cam.release()
            cv2.destroyAllWindows()
            ts = time.time()
            Date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            Time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            row = [Enrollment, Name, Date, Time]
            with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile, delimiter=',')
                writer.writerow(row)
                csvFile.close()
            res = "Images Saved for Enrollment : " + Enrollment + " Name : " + Name
            Notification.configure(text=res, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
            Notification.place(x=250, y=400)
        except FileExistsError as F:
            f = 'Student Data already exists'
            Notification.configure(text=f, bg="Red", width=21)
            Notification.place(x=450, y=400)


###for choose subject and fill attendance
def subjectchoose():
    def Fillattendances():
        sub=tx.get()
        now = time.time()  ###For calculate seconds of video
        future = now + 20
        if time.time() < future:
            if sub == '':
                err_screen1()
            else:
                recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
                try:
                    recognizer.read("TrainingImageLabel\Trainner.yml")
                except:
                    e = 'Model not found,Please train model'
                    Notifica.configure(text=e, bg="red", fg="black", width=33, font=('times', 15, 'bold'))
                    Notifica.place(x=20, y=250)

                harcascadePath = "haarcascade_frontalface_default.xml"
                faceCascade = cv2.CascadeClassifier(harcascadePath)
                df = pd.read_csv("StudentDetails\StudentDetails.csv")
                cam = cv2.VideoCapture(0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                col_names = ['Enrollment', 'Name', 'Date', 'Time']
                attendance = pd.DataFrame(columns=col_names)
                while True:
                    ret, im = cam.read()
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
                    for (x, y, w, h) in faces:
                        global Id

                        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                        if (conf <70):
                            print(conf)
                            global Subject
                            global aa
                            global date
                            global timeStamp
                            Subject = tx.get()
                            ts = time.time()
                            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            aa = df.loc[df['Enrollment'] == Id]['Name'].values
                            global tt
                            tt = str(Id) + "-" + aa
                            En = '15624031' + str(Id)
                            attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
                            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 7)
                            cv2.putText(im, str(tt), (x + h, y), font, 1, (255, 255, 0,), 4)

                        else:
                            Id = 'Unknown'
                            tt = str(Id)
                            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 25, 255), 7)
                            cv2.putText(im, str(tt), (x + h, y), font, 1, (0, 25, 255), 4)
                    if time.time() > future:
                        break

                    attendance = attendance.drop_duplicates(['Enrollment'], keep='first')
                    cv2.imshow('Filling attedance..', im)
                    key = cv2.waitKey(30) & 0xff
                    if key == 27:
                        break

                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                Hour, Minute, Second = timeStamp.split(":")
                fileName = "Attendance/" + Subject + "_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
                attendance = attendance.drop_duplicates(['Enrollment'], keep='first')
                print(attendance)
                attendance.to_csv(fileName, index=False)

                ##Create table for Attendance
                date_for_DB = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d')
                #REGISTER = str( Subject + "_" + date_for_DB + "_Time_" + Hour + "_" + Minute + "_" + Second)
                import psycopg2

        ###Connect to the database
                try:
                    global cursor
                    connection = psycopg2.connect(host="localhost",database="postgres",user="postgres",password="nickson1992")
                    cursor = connection.cursor()
                except Exception as e:
                    print(e)

                sql =  """CREATE TABLE IF NOT EXISTS REGISTER
                (ID INT NOT NULL  ,
                 ENROLLMENT varchar(100) NOT NULL,
                 NAME VARCHAR(50) NOT NULL,
                 DATE VARCHAR(20) NOT NULL,
                 TIME VARCHAR(20) NOT NULL,
                     PRIMARY KEY (ID)
                     );
                """
                ####Now enter attendance in Database
                insert_data =  "INSERT INTO  REGISTER  (ID,ENROLLMENT,NAME,DATE,TIME) VALUES (0, %s, %s, %s,%s)"
                VALUES = (str(Id), str(aa), str(date), str(timeStamp))
                try:
                    cursor.execute(sql)  ##for create a table
                    cursor.execute(insert_data, VALUES)##For insert data into table
                except Exception as ex:
                    print(ex)  #

                M = 'Attendance filled Successfully'
                Notifica.configure(text=M, bg="Green", fg="white", width=33, font=('times', 15, 'bold'))
                Notifica.place(x=20, y=250)

                cam.release()
                cv2.destroyAllWindows()

                import csv
                import tkinter
                root = tkinter.Tk()
                root.title("Attendance of " + Subject)
                root.configure(background='snow')
                cs = 'C:/Users/HP/Desktop/covid/Attendace managemnt system/' + fileName
                with open(cs, newline="") as file:
                    reader = csv.reader(file)
                    r = 0

                    for col in reader:
                        c = 0
                        for row in col:
                            # i've added some styling
                            label = tkinter.Label(root, width=8, height=1, fg="black", font=('times', 15, ' bold '),
                                                  bg="lawn green", text=row, relief=tkinter.RIDGE)
                            label.grid(row=r, column=c)
                            c += 1
                        r += 1
                root.mainloop()
                print(attendance)

    ###windo is frame for subject chooser
    windo = tk.Tk()
    windo.iconbitmap('AMS.ico')
    windo.title("Enter subject name...")
    windo.geometry('580x320')
    windo.configure(background='snow')
    Notifica = tk.Label(windo, text="Attendance filled Successfully", bg="Green", fg="white", width=33,
                            height=2, font=('times', 15, 'bold'))

    def Attf():
        import subprocess
        subprocess.Popen(r'explorer /select,"C:\Users\HP\Desktop\covid\Attendace_management_system-master\Attendace_management_system-master\Attendance\Manually Attendance\atttendance.csv"')

    attf = tk.Button(windo,  text="Check Sheets",command=Attf,fg="black"  ,bg="lawn green"  ,width=12  ,height=1 ,activebackground = "Red" ,font=('times', 14, ' bold '))
    attf.place(x=430, y=255)

    sub = tk.Label(windo, text="Enter Subject", width=15, height=2, fg="white", bg="blue2", font=('times', 15, ' bold '))
    sub.place(x=30, y=100)

    tx = tk.Entry(windo, width=20, bg="yellow", fg="red", font=('times', 23, ' bold '))
    tx.place(x=250, y=105)

    fill_a = tk.Button(windo, text="Fill Attendance", fg="white",command=Fillattendances, bg="deep pink", width=20, height=2,
                       activebackground="Red", font=('times', 15, ' bold '))
    fill_a.place(x=250, y=160)
    windo.mainloop()

def admin_panel():
    win = tk.Tk()
    win.iconbitmap('AMS.ico')
    win.title("LogIn")
    win.geometry('880x420')
    win.configure(background='snow')

    def log_in():
        username = un_entr.get()
        password = pw_entr.get()

        if username == 'nick' :
            if password == 'nickson1992':
                win.destroy()
                import tkinter
                import csv
                import pandas as pd
                
                root = Tk()
                root.geometry("500x200")
                root.title("Student Details")
                root.configure(background='snow')

                
                cs = ('C:/Users/HP/Desktop/covid/Attendace_management_system-master/Attendace_management_system-master/StudentDetails/StudentDetails.csv')
                
                
                #df = pd.read_csv(cs)
                with open(cs, newline="") as file:
                    reader = csv.reader(file)
                    r = 0

                    for col in reader:
                        c = 0
                        for row in col:
                            # i've added some styling
                            label = tkinter.Label(root, width=8, height=1, fg="black", font=('times', 15, ' bold '),
                                                  bg="lawn green", text=row, relief=tkinter.RIDGE)
                            label.grid(row=r, column=c)
                            c += 1
                        r += 1
                root.mainloop()
            else:
                valid = 'Incorrect ID or Password'
                Nt.configure(text=valid, bg="red", fg="black", width=38, font=('times', 19, 'bold'))
                Nt.place(x=120, y=350)

        else:
            valid ='Incorrect ID or Password'
            Nt.configure(text=valid, bg="red", fg="black", width=38, font=('times', 19, 'bold'))
            Nt.place(x=120, y=350)


    Nt = tk.Label(win, text="Attendance filled Successfully", bg="Green", fg="white", width=40,
                  height=2, font=('times', 19, 'bold'))
    # Nt.place(x=120, y=350)

    un = tk.Label(win, text="Enter username", width=15, height=2, fg="white", bg="blue2",
                   font=('times', 15, ' bold '))
    un.place(x=30, y=50)

    pw = tk.Label(win, text="Enter password", width=15, height=2, fg="white", bg="blue2",
                  font=('times', 15, ' bold '))
    pw.place(x=30, y=150)

    def c00():
        un_entr.delete(first=0, last=22)

    un_entr = tk.Entry(win, width=20, bg="yellow", fg="red", font=('times', 23, ' bold '))
    un_entr.place(x=290, y=55)

    def c11():
        pw_entr.delete(first=0, last=22)

    pw_entr = tk.Entry(win, width=20,show="*", bg="yellow", fg="red", font=('times', 23, ' bold '))
    pw_entr.place(x=290, y=155)

    c0 = tk.Button(win, text="Clear", command=c00, fg="black", bg="deep pink", width=10, height=1,
                            activebackground="Red", font=('times', 15, ' bold '))
    c0.place(x=690, y=55)

    c1 = tk.Button(win, text="Clear", command=c11, fg="black", bg="deep pink", width=10, height=1,
                   activebackground="Red", font=('times', 15, ' bold '))
    c1.place(x=690, y=155)

    Login = tk.Button(win, text="LogIn", fg="black", bg="lime green", width=20,
                       height=2,
                       activebackground="Red",command=log_in, font=('times', 15, ' bold '))
    Login.place(x=290, y=250)
    win.mainloop()


###For train the model
def trainimg():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    global detector
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    try:
        global faces,Id
        faces, Id = getImagesAndLabels("TrainingImages")
    except Exception as e:
        l='please make "TrainingImage" folder & put Images'
        Notification.configure(text=l, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
        Notification.place(x=350, y=400)

    recognizer.train(faces, np.array(Id))
    try:
        recognizer.save("TrainingImageLabel\Trainner.yml")
    except Exception as e:
        q='Please make "TrainingImageLabel" folder'
        Notification.configure(text=q, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
        Notification.place(x=350, y=400)

    res = "Model Trained"  # +",".join(str(f) for f in Id)
    Notification.configure(text=res, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
    Notification.place(x=250, y=400)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)
        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids


window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
window.iconbitmap('AMS.ico')
def mask_recog():
    x='C:/Users/HP/Desktop/covid/facemask/caffe/facede/models/deploy.prototxt.txt'
    y='C:/Users/HP/Desktop/covid/facemask/caffe/facede/models/res10_300x300_ssd_iter_140000.caffemodel'

#Load Model
    print("Loading model...................")
    net = cv2.dnn.readNetFromCaffe(x,y)
    model=load_model('C:/Users/HP/mask_recog_ver2.h5')
# initialize the video stream to get the video frames
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

#loop the frams from the  VideoStream
    while True :
    #Get the frams from the video stream and resize to 400 px
        frame = vs.read()
        frame = imutils.resize(frame,width=400)

    # extract the dimensions , Resize image into 300x300 and converting image into blobFromImage
        (h, w) = frame.shape[:2]
    # blobImage convert RGB (104.0, 177.0, 123.0)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # passing blob through the network to detect and pridiction
        net.setInput(blob)
        detections = net.forward()

    # loop over the detections
        for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with
	# the detection
            confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the confidence is
	# greater than the minimum confidence
            if confidence > 0.3:
		# compute the (x, y)-coordinates of the bounding box for
		# the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
		# ensure the bounding boxes fall within the dimensions of
		# the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        # extract the face ROI, convert it from BGR to RGB channel
		# ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
		# pass the face through the model to determine if the face
		# has a mask or not
                (mask, withoutMask) = model.predict(face)[0]# determine the class label and color we'll use to draw
		# the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		# include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# display the label and bounding box rectangle on the output
		# frame
                cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
            if key == ord("q"):
             break

# do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
def social_distance_detection():
    MIN_CONF = 0.3    # minimum object detection confidence
    NMS_THRESH = 0.3  # non-maxima suppression threshold

# boolean indicating if NVIDIA CUDA GPU should be used
    USE_GPU = False

# define the minimum safe distance (in pixels) that two people can be
# from each other
    MIN_DISTANCE = 50
    def detect_people(frame, net, ln, personIdx=0):
        # grab the dimensions of the frame and  initialize the list of
    # results
        (H, W) = frame.shape[:2]
        results = []

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, centroids, and
    # confidences, respectively
        boxes = []
        centroids = []
        confidences = []

    # loop over each of the layer outputs
        for output in layerOutputs:
        # loop over each of the detections  
            for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
            
            # filter detections by (1) ensuring that the object
            # detected was a person and (2) that the minimum
            # confidence is met
                if classID == personIdx and confidence > MIN_CONF:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                # update our list of bounding box coordinates,
                # centroids, and confidences
                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))
    
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
    
    # ensure at least one detection exists
        if len(idxs) > 0:
        # loop over the indexes we are keeping
            for i in idxs.flatten():
            # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
            
            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
                r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(r)

    # return the list of results
        return results
        # derive the paths to the YOLO weights and model configuration
    weightsPath = 'C:/Users/HP/Downloads/yolov3.weights'
    configPath = 'C:/Users/HP/Desktop/covid/facemask/yolov3/cfg/yolov3.cfg'
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    labelsPath = 'C:/Users/HP/Desktop/covid/facemask/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    #print(LABELS[0])
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def video_stream():
        print("[INFO] accessing video stream...")
        cap = cv2.VideoCapture('C:/Users/HP/Desktop/tensorflow/Social_Distancing-CV-master/Social_Distancing-CV-master/people.mp4')
        writer = None
        while True:
            # read the next frame from the file
            (grabbed, frame) = cap.read()
            # if the frame was not grabbed, then we have reached the end
            if not grabbed:
                break
            # resize the frame and then detect people (and only people) in it
            frame = imutils.resize(frame, width=700)
            results = detect_people(frame, net, ln,
                            personIdx=LABELS.index("person"))

            # initialize the set of indexes that violate the minimum social
    # distance
            violate = set()
            # ensure there are *at least* two people detections (required in
    # order to compute our pairwise distance maps)
            if len(results) >= 2:
                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")
                # loop over the upper triangular of the distance matrix
                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                # centroid pairs is less than the configured number
                # of pixels
                        if D[i, j] < MIN_DISTANCE:
                    # update our violation set with the indexes of
                    # the centroid pairs
                            violate.add(i)
                            violate.add(j)
            for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
        # initialize the color of the annotation
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)
                # if the index pair exists within the violation set, then
        # update the color
                if i in violate: 
                    color = (0, 0, 255)
        # draw (1) a bounding box around the person and (2) the
        # centroid coordinates of the person,
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)

         # draw the total number of social distancing violations on the
    # output frame
            text = "Social Distancing Violations: {}".format(len(violate))
            cv2.putText(frame, text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
                    # check to see if the output frame should be displayed to our
    # screen
            cv2.imshow("Frame", frame)
    
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    cv2.destroyAllWindows()
    

    dow = tk.Tk()
    dow.title("CMS-Covid Management System")

    dow.geometry('720x400')
    dow.configure(background='cyan')
    video_stream = tk.Button(dow, text="live stream",command=video_stream,fg="white"  ,bg="blue2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
    video_stream.place(x=90, y=50)

    analysis = tk.Button(dow, text="social distance analysis",fg="black",command=trainimg ,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
    analysis.place(x=390, y=50)

    btn.grid(column=2, row=0)
    dow.mainloop()


def on_closing():
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()
window.protocol("WM_DELETE_WINDOW", on_closing)

message = tk.Label(window, text="MMUST-COVID-19-MANAGEMENT-SYSTEM", bg="cyan", fg="black", width=50,
                   height=3, font=('times', 30, 'italic bold '))

message.place(x=80, y=20)

Notification = tk.Label(window, text="All things good", bg="Green", fg="white", width=15,
                      height=3, font=('times', 17, 'bold'))

lbl = tk.Label(window, text="Enter Enrollment", width=20, height=2, fg="black", bg="deep pink", font=('times', 15, ' bold '))
lbl.place(x=200, y=200)

def testVal(inStr,acttyp):
    if acttyp == '1': #insert
        if not inStr.isdigit():
            return False
    return True

txt = tk.Entry(window, validate="key", width=20, bg="yellow", fg="red", font=('times', 25, ' bold '))
txt['validatecommand'] = (txt.register(testVal),'%P','%d')
txt.place(x=550, y=210)

lbl2 = tk.Label(window, text="Enter Name", width=20, fg="black", bg="deep pink", height=2, font=('times', 15, ' bold '))
lbl2.place(x=200, y=300)

txt2 = tk.Entry(window, width=20, bg="yellow", fg="red", font=('times', 25, ' bold '))
txt2.place(x=550, y=310)

clearButton = tk.Button(window, text="Clear",command=clear,fg="black"  ,bg="deep pink"  ,width=10  ,height=1 ,activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton.place(x=950, y=210)

clearButton1 = tk.Button(window, text="Clear",command=clear1,fg="black"  ,bg="deep pink"  ,width=10 ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton1.place(x=950, y=310)

maskdetection = tk.Button(window, text="Detect Mask",command=mask_recog,fg="black"  ,bg="cyan"  ,width=20 ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
maskdetection.place(x=90, y=600)
sdistance = tk.Button(window, text="Social Distance Detection",command=social_distance_detection,fg="black"  ,bg="cyan"  ,width=20 ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
sdistance.place(x=390, y=600)
AP = tk.Button(window, text="Check Register students",command=admin_panel,fg="black"  ,bg="cyan"  ,width=19 ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
AP.place(x=990, y=410)

takeImg = tk.Button(window, text="Take Images",command=take_img,fg="white"  ,bg="blue2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=90, y=500)

trainImg = tk.Button(window, text="Train Images",fg="black",command=trainimg ,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=390, y=500)

FA = tk.Button(window, text="Automatic Attendace",fg="white",command=subjectchoose  ,bg="blue2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
FA.place(x=690, y=500)

quitWindow = tk.Button(window, text="Manually Fill Attendance", command=manually_fill  ,fg="black"  ,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=990, y=500)

window.mainloop()