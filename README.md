##  Covid-19 Management system using Face👦🏻👧 Recognition [![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg)]
### Screenshots

### Basic UI
<img src="https://github.com/Nickjas/covid-19-management-system//blob/gh-pages/sc1.PNG">

### When it's Recognise me
<img src="https://github.com/Nickjas/covid-19-management-system//blob/gh-pages/sc1.PNG">

### When it's fill a attendace
<img src="https://github.com/Nickjas/covid-19-management-system/screenshoot(43).PNG">

### determine whether one has a mask or not
<img src="https://github.com/Nickjas/covid-19-management-system/blob/gh-pages/screnshot(41).PNG">


### How it works? See:)

<img src="https://github.com/Nickjas/covid-19-management-system/AMS.gif">

### Code Requirements
- Opencv(`pip install opencv-python`)
- Tkinter(Available in python)
- PIL (`pip install Pillow`)
- Pandas(`pip install pandas`)
- Tensorflow
- Matplotlib
- MTCNN
- imutils


### What steps you have to follow??
- Download my Repository 
- Create a `TrainingImage` folder in a project.
- Open a `cms.py` and change the all paths with your system path
- Run `cms.py`.

### Project Structure

- After run you need to give your face data to system so enter your ID and name in box than click on `Take Images` button.
- It will collect 200 images of your faces, it save a images in `TrainingImage` folder
- After that we need to train a model(for train a model click on `Train Image` button.
- It will take 5-10 minutes for training(for 10 person data).
- After training click on `Automatic Attendance` ,it can fill attendace by your face using our trained model (model will save in `TrainingImageLabel` )
- it will create `.csv` file of attendance according to time & subject.
- You can store data in database,change the DB name according to your in `AMS_Run.py`.
- `Manually Fill Attendace` Button in UI is for fill a manually attendance (without face recognition),it's also create a `.csv` and store in a database.
-'Face mask detection' determines whether one has a mask or not
-social distance button checks the distance between people






### Notes
- It will require high processing power(I have 8 GB RAM & 2 GB GC)
- If you think it will recognise person just like humans,than leave it ,its not possible.
- Noisy image can reduce your accuracy so quality of images matter.

## Just follow☝️ me and Star⭐ my repository 

