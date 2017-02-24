
# coding: utf-8

# In[139]:

#Load nessassary modules
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math
import cmath
import PIL
from PIL import ImageDraw
import PIL.ImageTk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from matplotlib import colors
from pylab import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from os import path
get_ipython().magic('matplotlib inline')


# In[140]:

#The mathematical functions nessassary to find the uv track from the array parameters
class interferometer:
    
    #Convert single point coordinates into ENU
    def ENU(coords):
        arrayInQuestion = list(np.zeros([len(coords[0]),3]))
        arrayInQuestion2 = list(np.zeros([len(arrayInQuestion[0]),len(arrayInQuestion)]))
        for i in np.arange(len(coords)):
            E = -coords[i][0]*np.cos(coords[i][1])*np.sin(coords[i][2])
            N = coords[i][0]*np.cos(coords[i][1])*np.cos(coords[i][2])
            U = coords[i][0]*np.sin(coords[i][1])
            
            arrayInQuestion[i] = [E,N,U]
            arrayInQuestion2[i] = arrayInQuestion[:][i]
        return arrayInQuestion2
    
    #Convert ENU to xyz, where L is the latitude
    def ENUtoUVW(coords1,coords2,L,lam,delta,h):
        Bx = coords2[0]-coords1[0]
        By = coords2[1]-coords1[1]
        Bz = coords2[2]-coords1[2]
        u = (np.sin(h)*(-np.sin(L)*By+np.cos(L)*Bz)+np.cos(h)*(Bx))/(lam*1000)
        v = (-np.sin(delta)*np.cos(h)*(-np.sin(L)*By+np.cos(L)*Bz)+np.sin(delta)*np.sin(h)*(Bx)+np.cos(delta)*np.cos(L)*By+np.sin(L)*Bz)/(lam*1000)
        w = (np.cos(delta)*np.cos(h)*(-np.sin(L)*By+np.cos(L)*Bz)-np.cos(delta)*np.sin(h)*(Bx)+np.sin(delta)*np.cos(L)*By+np.sin(L)*Bz)/(lam*1000)
        return np.array([u,v,w])

    #Calculates the hour vector of the track based on the hour angles and sample rate
    def timetrackHours(hCen,hStep,hSteps):
        return np.linspace(hCen-hStep*hSteps/2,hCen+hStep*hSteps/2,hSteps+1)
    
    #Find uv track over time, L is lattitude, delta is declination, hArray are all the hour angle values
    def timetrack(coords,L,lam,delta,hArray): #H is taken as an array
        uvFu = np.array([])
        uvFv = np.array([])
        uvRu = np.array([])
        uvRv = np.array([])
        uvFu = np.array(list(interferometer.uvTrack(coords,L,lam,delta,hArray)[:,0,:]))
        uvFv = np.array(list(interferometer.uvTrack(coords,L,lam,delta,hArray)[:,1,:])) 
        for i in np.arange(len(hArray)):
            uvRu = np.array(list(uvRu)+list(uvFu[:,i]))
            uvRv = np.array(list(uvRv)+list(uvFv[:,i]))
        return np.array([uvRu,uvRv])
    
    #Create all the baseline uv vectors, L is latitude, lam is wavelength, delta is declination, h is hour angle (h is an array)
    def uvTrack(coords,L,lam,delta,h):
        uv = np.zeros([len(coords[1])*(len(coords[1])-1),3,len(h)])
        k = 0
        for i in np.arange(len(coords[1])):
            for j in np.arange(len(coords[1])-1-i):
                halfTrack = interferometer.ENUtoUVW(np.array([coords[0][i],coords[1][i],coords[2][i]]),np.array([coords[0][i+j+1],coords[1][i+j+1],coords[2][i+j+1]]),L,lam,delta,h)
                uv[k] = halfTrack
                uv[k+1] = -halfTrack
                k = k+2
        return np.array(uv)
    
    #Scale the UV track to the Fourier transformed image, uv cover is the track to be scaled, res is the pixel scale in arcseconds and image is the image to scale too (so the total size is known)
    def ScaleUVTrack(uvCover,Res,image):
        resRad = np.radians(Res/(60.*60.))
        #sizeAng = (resRad)*np.array([image.size[0],image.size[1]])
        inversedSizeAng = 2./resRad
        pixSize = np.array([inversedSizeAng/len(np.fft.fft2(image)),inversedSizeAng/len(np.fft.fft2(image)[0])])
        return 1000.*np.array([uvCover[0]/pixSize[0],uvCover[1]/pixSize[1]])
    
    #Take the created UV track, then make the mask scaled to the pixel size for the image, ima is the image used (to determine the size of the track), uv track is the track to create mask from, res is the pixel scale in arcseconds 
    def makeMask(ima,uvCover,Res):
    
        #set up uv track for use in mask
        finWid = len(np.fft.fft2(ima))
        finLen = len(np.fft.fft2(ima)[0])
        sampleUsed = interferometer.ScaleUVTrack(uvCover,Res,ima)
        sampleUsed2 = np.zeros([len(sampleUsed[0]),len(sampleUsed)])
        for i in np.arange(len(sampleUsed[0])):
            sampleUsed2[i] = sampleUsed[:,i]
            sampleUsed2[i,1] = -sampleUsed2[i,1]
    
        #create black image to imprint uv track on
        track = PIL.Image.new('RGB', (finLen,finWid), 'black')
        mask = ImageDraw.Draw(track)
        coord = sampleUsed2
        pointSize = 1
        Coord = np.arange(50)-25
        Coord2 = np.zeros([len(Coord)**2,2])
        m = 0
        
        #place a white point at the points of the mask corrisponding to the points of the uv track
        for i in np.arange(len(Coord)):
            for j in np.arange(len(Coord)):
                Coord2[m,0] = Coord[i]
                Coord2[m,1] = Coord[j]
                m+=1
        for (x,y) in coord:
            mask.rectangle([x+finLen/2,y+finWid/2,x+finLen/2+pointSize-1,y+finWid/2+pointSize-1], fill = 'white')
        #mask.rectangle([0,0,1000,1000], fill = 'white') #Use this for a full image
        return track
    
    def observedFFT(img,uvTrack):
        imUsed = np.fft.fftshift(np.fft.fft2(img))
        trackF = np.zeros([uvTrack.size[1],uvTrack.size[0]])
        trackF = imUsed*np.asarray(uvTrack)[:,:]/255
        return trackF
    
    def synthesisedBeam(ima,uvCover,Res):
        #this function find's the synthesised beam on an image from a given set of uv coordinates
        #it is not used in the orignal tk interface since the uvtrack was already found, and it would be a waste of time to find again
        #however it is avaliable if need
        track = interferometer.makeMask(ima,uvCover,Res)
        return abs(np.fft.fftshift(np.fft.ifft2(track.convert('L'))))
    
    def observedImage(img, uvCover, Res):
        #once again, not used in the orignal code for time saving reasons, but avaliabe if other steps are not taken
        track = interferometer.makeMask(img,uvCover,Res).convert('1')
        trackAndFFT =interferometer.observedFFT(img,track)
        return np.fft.ifft2(np.fft.ifftshift(trackAndFFT))


# In[141]:

#This class defines the frames used to display the nessassary images in the output section of the code
class ImageFrame:
    
    def __init__ (self, parent,image=PIL.Image.new('RGB', (1,1), 'white'), width=250, height=250, text="Unamed", relief=SUNKEN):
        
        #Initialise Variables
        self.parent = parent #sets the parent to this widget
        self.image = image #the image used in the widget
        self.width = width #The width of the frame, field of view of the image
        self.height = height #The height of the frame
        self.imput = text #The title of the frame
        self.rel = relief #the relief of the surronding frames
        self.PosX = self.width/2 #the position of the image, starts in the centre but can be dragged
        self.PosY = self.height/2 #as above
        self.imSizeX = self.width #the size of the image in the frame, starts as the image size but can be zoomed
        self.imSizeY = self.height#as above
        self.oldMouseX = 0#The position of the mouse, tracks how it moves
        self.oldMouseY = 0#As above
        
        #Set up the image to be displayed
        self.frame = LabelFrame(self.parent,width=self.width,height=self.height,text=self.imput,relief=self.rel)
        self.can = Canvas(self.frame,width=self.width, height=self.height)
        self.imSized = self.image.resize([self.imSizeX,self.imSizeY])
        self.imShowReady = PIL.ImageTk.PhotoImage(self.imSized, master = self.can)
        self.can.create_image(self.PosX,self.PosY,image=self.imShowReady)
        self.can.grid(row=0,column=0)
        
        #Bind the zoom and motion comands to the image
        self.can.bind('<Button - 1>', lambda e: self.press(e)) #Find intial position maps to pressing left mouse button
        self.can.bind('<B1-Motion>', lambda e: self.pull(e)) #Dragging view maps to dragging left mouse
        self.can.bind('<MouseWheel>', self.zoomIn) #Zoom mapped to mouse wheel
    
    #This finds the intial mouse position when you click on the image
    def press(self, mo):
        self.oldMouseX = mo.x
        self.oldMouseY = mo.y
    
    #Drags the field of view with the mouse based on the change in position
    def pull(self, mo):
        self.PosX += mo.x-self.oldMouseX
        self.PosY += mo.y-self.oldMouseY
        self.oldMouseX = mo.x
        self.oldMouseY = mo.y
        
        self.imSized = self.image.resize([int(self.imSizeX),int(self.imSizeY)])
        self.imShowReady = PIL.ImageTk.PhotoImage(self.imSized, master = self.can)
        self.can.create_image(self.PosX,self.PosY,image=self.imShowReady)
    
    #Zooms the view in and out
    def zoomIn(self,event):
        if (event.delta == 120): #Zoom in
            self.imSizeX += 50
            self.imSizeY += 50*self.height/self.width
        if (event.delta == -120): #Zoom out
            self.imSizeX -= 50
            self.imSizeY -= 50*self.height/self.width
        
        self.imSized = self.image.resize([int(self.imSizeX),int(self.imSizeY)])
        self.imShowReady = PIL.ImageTk.PhotoImage(self.imSized, master = self.can)
        self.can.create_image(self.PosX,self.PosY,image=self.imShowReady)
    
    #This function can be called later to turn the blank frame into an imaged one
    def addImage (self, image):
        self.image = image
        
        self.imSized = self.image.resize([int(self.imSizeX),int(self.imSizeY)])
        self.imShowReady = PIL.ImageTk.PhotoImage(self.imSized, master = self.can)
        self.can.create_image(self.PosX,self.PosY,image=self.imShowReady)


# In[142]:

#This wiget allows you select an array to be added to the configurations
class ArraySelecter:
    
    #Initialize everything
    def __init__(self, parent, displayHeight=4, displayWidth=4, configurationsPOLAR=[], configurationsENU=[]):
        #Initialize Variables
        self.parent = parent #The Parent
        self.displayHeight = displayHeight #The Height of the displayed plot
        self.displayWidth = displayWidth #The Width of the displayed plot
        self.configurationsNames = [] #Will be the names of the various configurations
        self.configurationsCoords = [] #Will be a list of the array configurations
        self.configurationsLat = [] #Will be a list of the array's lattitudes
        
        #The next IF statements are to insert said names and array configurations
        if (configurationsPOLAR != []):
            self.configurationsNames = self.configurationsNames+configurationsPOLAR[0]
            self.configurationsCoords = self.configurationsCoords+interferometer.ENU(configurationsPOLAR[1])
            self.configurationsLat = self.configurationsLat+configurationsPOLAR[2]
        if (configurationsENU != []):
            self.configurationsNames = self.configurationsNames+configurationsENU[0]
            self.configurationsCoords = self.configurationsCoords+configurationsENU[1]
            self.configurationsLat = self.configurationsLat+configurationsENU[2]
        self.configurations = [self.configurationsNames, self.configurationsCoords, self.configurationsLat] #condence everything to into a single list
        
        #set up the frame, and label
        self.frame = Frame(self.parent)
        
        ttk.Label(self.frame, text="Select Array Configurations").grid(column=1,row=1)
        
        #The list box, with the ability to change the diplay of the plot
        self.configList = Listbox(self.frame)
        for i in np.arange(len(self.configurations[0])):
            self.configList.insert(END,self.configurations[0][i])
        self.configList.grid(column=1, row = 2)
        
        self.scrollerArrays = ttk.Scrollbar(self.frame, command = self.configList.yview)
        self.scrollerArrays.grid(column=1,row=2,rowspan=3,sticky=(E,N,S))
        self.configList.configure(yscrollcommand=self.scrollerArrays.set)
        
        #Canvas that displays the plot
        rcParams['figure.figsize'] = 4, 4
        self.arrDisFig = plt.figure(1)
        self.Ea = self.configurations[1][0][0]
        self.No = self.configurations[1][0][1]
        plt.xlabel('East(km)')
        plt.ylabel('North(km)')
        plt.plot(self.Ea/1000,self.No/1000,'o',markersize=1)
        
        self.arrayDisplay = FigureCanvasTkAgg(self.arrDisFig,master = self.frame)

        self.plotDisplay = self.arrayDisplay.get_tk_widget()
        self.plotDisplay.grid(column=2,row=1,rowspan=10)
        
        #Establish the change in the displayed array when the left mouse is released
        self.configList.bind('<ButtonRelease - 1>', self.ChooseArray)
    
    #replots the displayed array
    def ChooseArray(self,event):
        rcParams['figure.figsize'] = 4, 4
        self.arrDisFig = plt.figure(1)
        self.Ea = self.configurations[1][self.configList.curselection()[0]][0]
        self.No = self.configurations[1][self.configList.curselection()[0]][1]
        plt.clf()
        plt.xlabel('East(km)')
        plt.ylabel('North(km)')
        plt.plot(self.Ea/1000,self.No/1000,'o',markersize=1)
        
        self.arrayDisplay = FigureCanvasTkAgg(self.arrDisFig,master = self.frame)
        
        self.plotDisplay.grid_forget()
        self.plotDisplay = self.arrayDisplay.get_tk_widget()
        self.plotDisplay.grid(column=2,row=1,rowspan=10)


# In[143]:

#This widget handles the directly imported parameters (sample frequency, light frequency and declination), all off these are done via direct entry
class Parameters:
    
    def __init__(self, parent, frequency=0, sampleTime=0, declination=0):
        self.parent = parent #Set the parent
        self.freq = StringVar() #Variable for the parent
        self.freq.set(frequency) #Initialize
        self.sampleTime = StringVar() #Variable for the sample rate in seconds
        self.sampleTime.set(sampleTime)
        self.dec = StringVar() #Variable for the declination
        self.dec.set(declination)
        
        self.frame = Frame(self.parent)

        ttk.Label(self.frame, text="Set Parameters").grid(column=1, row=1, columnspan=2, sticky=N)
        
        #Labels the variables
        ttk.Label(self.frame,text='Frequency (MHz):').grid(column=1,row=2)
        ttk.Label(self.frame,text='Sample time(sec):').grid(column=1,row=3)
        ttk.Label(self.frame,text='Declination (deg):').grid(column=1,row=4)
        
        #Allow modification of the variables
        ttk.Entry(self.frame,textvariable= self.freq).grid(column=2,row=2)
        ttk.Entry(self.frame,textvariable=self.sampleTime).grid(column=2,row=3)
        ttk.Entry(self.frame,textvariable=self.dec).grid(column=2,row=4)


# In[144]:

#This widget creates the hour angle slider and the ability to manipulate the values directly
class HourSlider:
    
    def __init__(self, parent,minHour=0,maxHour=0, length = 100):
        self.parent = parent #set variable for the parent
        self.minHour = StringVar() #Set the variable for the smallest hour angle
        self.minHour.set(minHour)
        self.maxHour = StringVar() #Set the variable for the smallest hour angle
        self.maxHour.set(maxHour)
        self.length = length #Length of the bar
        self.minHoVal = 10 #Initialises the position of the sliders
        self.maxHoVal = self.length+10 
        self.hourMove = 0 #sets the current mode of the slider (not clicked on)
        self.hourMinPrediction = False #determines wheather clicking on the hour bar will guess
        
        self.frame = Frame(self.parent)
        
        ttk.Label(self.frame, text = "Hour Angle:").grid(column=1,row=1)

        #Sets up the hour slider
        self.hourCanvas = Canvas(self.frame, width=self.length+20,height=50)
        self.hourCanvas.grid(column=1,row=2,columnspan=2)
        self.hourCanvas.create_line(5,25,self.length+5,25)
        self.hourCanvas.create_rectangle(self.minHoVal-5,10,self.minHoVal+5,40,fill='white',outline='black')
        self.hourCanvas.create_rectangle(self.maxHoVal-5,10,self.maxHoVal+5,40,fill='white',outline='black')
        
        #Formulation for the hour angle as a result of the sliders position
        self.minHour.set((self.minHoVal-10)/self.length*24-12)
        self.maxHour.set((self.maxHoVal-10)/self.length*24-12)
        
        #Entry to set min hour
        self.minHoLabel = ttk.Entry(self.frame, textvariable = self.minHour)
        self.minHoLabel.grid(column=1,row=3)

        #Allows the slider to respond to the entries
        self.frame.bind('<Enter>', self.hourDirect)

        #Entry to set max hour
        self.maxHoLabel = ttk.Entry(self.frame, textvariable = self.maxHour)
        self.maxHoLabel.grid(column=2,row=3)

        #Events to the slider
        self.hourCanvas.bind('<1>', lambda e: self.GrabHourSlide(e))
        self.hourCanvas.bind('<ButtonRelease - 1>',lambda e: self.ReleaseHour)
        self.hourCanvas.bind('<B1-Motion>', lambda e: self.Slide(e))
        
    #Used to set the hour angle directly, this activates when the mouse moves into the slider canvas
    def hourDirect(self, event):
        #Ensure that the hour angle with within +12 and -12 hours
        if ((float(self.minHour.get()) <= 12) and (float(self.minHour.get()) >= -12) and (float(self.maxHour.get()) <= 12) and (float(self.maxHour.get()) >= -12)):
            if ((float(self.minHour.get()) > float(self.maxHour.get()))):
                self.maxHour.set(self.minHour.get())
        
            self.minHoVal = (float(self.minHour.get())+12)*self.length/24+10
            self.maxHoVal = (float(self.maxHour.get())+12)*self.length/24+10
            #Update canvas
            self.hourCanvas.delete('all')
            self.hourCanvas.create_line(5,25,self.length+5,25)
            self.hourCanvas.create_rectangle(self.minHoVal-5,10,self.minHoVal+5,40,fill='white',outline='black')
            self.hourCanvas.create_rectangle(self.maxHoVal-5,10,self.maxHoVal+5,40,fill='white',outline='black')

            self.hourCanvas.bind('<1>', lambda e: self.GrabHourSlide(e))
            self.hourCanvas.bind('<ButtonRelease - 1>',lambda e: self.ReleaseHour)
            self.hourCanvas.bind('<B1-Motion>', lambda e: self.Slide(e))
        else:
            self.minHour.set(-12)
            self.maxHour.set(12)
            
    #Move the hour canvas slider when selected
    def Slide(self,e):
            
        #Determine the position of the slider
        if (self.hourMove == 1):
            self.minHoVal = e.x
            if (self.maxHoVal < self.minHoVal):
                self.maxHoVal = self.minHoVal
        if (self.hourMove == 2):
            self.maxHoVal = e.x
            if (self.maxHoVal < self.minHoVal):
                self.minHoVal = self.maxHoVal
        if (self.minHoVal < 10):
            self.minHoVal = 10
        if (self.maxHoVal < 10):
            self.minHoVal = 10
            self.maxHoVal = 10
        if (self.maxHoVal > self.length+10):
            self.maxHoVal = self.length+10
        if (self.minHoVal > self.length+10):
            self.minHoVal = self.length+10
            self.maxHoVal = self.length+10
        
        #Set display values appropriately for the slider
        if (self.hourMove != 0):
            self.minHour.set("%.2f" %((self.minHoVal-10)/self.length*24-12))
            self.maxHour.set("%.2f" %((self.maxHoVal-10)/self.length*24-12))
        
            #Update slider canvas with new positons
            self.hourCanvas.delete('all')
            self.hourCanvas.create_line(5,25,self.length+5,25)
            self.hourCanvas.create_rectangle(self.minHoVal-5,10,self.minHoVal+5,40,fill='white',outline='black')
            self.hourCanvas.create_rectangle(self.maxHoVal-5,10,self.maxHoVal+5,40,fill='white',outline='black')

            self.hourCanvas.bind('<1>', lambda e: self.GrabHourSlide(e))
            self.hourCanvas.bind('<ButtonRelease - 1>',lambda e: self.ReleaseHour)
            self.hourCanvas.bind('<B1-Motion>', lambda e: self.Slide(e))
    
    #release the hour slider when the mouse is released
    def ReleaseHour(self):
        self.hourMove = 0
        
    #Used to detect if you mouse curser is in the right position to grab one of the slider bars on the hour selecter
    def GrabHourSlide(self,e):
        if ((e.x < self.minHoVal+5) and (e.x > self.minHoVal-5)):
            self.hourMove = 1
            self.hourMinPrediction = True
        else:
            if ((e.x < self.maxHoVal+5) and (e.x > self.maxHoVal-5)):
                self.hourMove = 2
                self.hourMaxPrediction = False
            else:
                if (self.hourMinPrediction == False):
                    self.hourMove = 1
                    self.Slide(e)
                    self.hourMinPrediction = True
                else:
                    self.hourMove = 2
                    self.Slide(e)
                    self.hourMinPrediction = False


# In[145]:

class AddButton:
    #The button to add an array configuration to the selected configurations and display it's uv track
    def __init__(self,parent, text='Unnamed'):
        self.parent = parent
        self.text = text
        
        self.frame = Frame(parent)
        
        #The button
        self.add = ttk.Button(self.frame, text = self.text, command = self.AddSelected)
        self.add.grid(column=1,row=1, columnspan = 1, sticky = N)
    
    #Add a selected array configuration to the arrays to use for the formulations
    def AddSelected(self):
        #add array name to box with the used arrays on it
        uvDisplay.originalIndex.append(arrayBox.configList.curselection()[0])
        uvDisplay.usedArrays.insert(END, arrayBox.configurations[0][arrayBox.configList.curselection()[0]] + ':      ' + "%.2f" %float(hourSlide.minHour.get()) + '-' + "%.2f" %float(hourSlide.maxHour.get()))
        self.hour = (float(hourSlide.minHour.get())/2.+float(hourSlide.maxHour.get())/2.)*np.pi/180
        uvDisplay.relationHours.append(interferometer.timetrackHours(self.hour,2.*0.99726958*np.pi/(24.*60.*60./float(params.sampleTime.get())),(float(hourSlide.maxHour.get())-float(hourSlide.minHour.get()))*60.*60./float(params.sampleTime.get())/0.99726958))
        uvDisplay.arrays.append(interferometer.timetrack(arrayBox.configurations[1][arrayBox.configList.curselection()[0]],arrayBox.configurations[2][arrayBox.configList.curselection()[0]],299792458/(float(params.freq.get())*10**6),float(params.dec.get())*np.pi/180,uvDisplay.relationHours[len(uvDisplay.relationHours)-1]))


# In[156]:

#This wiget provides  a list of the arrays selected as part of the configurations, and also displays the uv track, as well as the remove button
class SelectedDisplay:
    
    def __init__(self,parent,displayHeight=4, displayWidth=4, configurations=[]):
        self.parent = parent #The parent of the widget
        self.disHeight = displayHeight #Ahhh, I'll finish commenting later
        self.disWidth = displayWidth
        self.arrays = configurations
        self.relationHours = []
        self.originalIndex = []
        self.lastFreq = 0
        self.lastSampleTime = 0
        self.lastdec = 0
        
        self.frame = Frame(self.parent)
        
        ttk.Label(self.frame, text = "Current Configurations").grid(column=1,row=1)

        self.usedArrays = Listbox(self.frame)
        self.usedArrays.grid(column=1,row=2, rowspan=3)

        self.scrollerUV = ttk.Scrollbar(self.frame, command = self.usedArrays.yview)
        self.scrollerUV.grid(column=1,row=2,rowspan=3,sticky=(E,N,S))
        self.usedArrays.configure(yscrollcommand=self.scrollerUV.set)
        
        self.delete = ttk.Button(self.frame, text = "Delete Selected", command = self.DeleteSelected)
        self.delete.grid(column=1,row=5)
        
        self.delete = ttk.Button(self.frame, text = "Plot UV", command = self.plotUV)
        self.delete.grid(column=1,row=6)
    
    def replot(self):
        self.display = Toplevel(root)
        self.display.wm_title("Window")
        self.uvDisFig = plt.figure(2)
        plt.clf()
        plt.xlabel('U(klam)')
        plt.ylabel('V(klam)')
        for i in np.arange(len(self.arrays)):
            self.U = self.arrays[i][0]
            self.V = self.arrays[i][1]
            plt.plot(self.U,self.V,'o',markersize=0.1)
        
        self.uvDis = FigureCanvasTkAgg(self.uvDisFig,master = self.display)
    
        self.scrollerUV.grid_forget()
        self.scrollerUV = ttk.Scrollbar(self.frame, command = self.usedArrays.yview)
        self.scrollerUV.grid(column=1,row=2,rowspan=3,sticky=(E,N,S))
        self.usedArrays.configure(yscrollcommand=self.scrollerUV.set)
        self.plotUVDis = self.uvDis.get_tk_widget()
        
    #remove a selected array from the chosen configurations (so it will not be used in final calculations)
    def DeleteSelected(self):
        #get the arrays being used, remove the selected
        self.arrays.pop(self.usedArrays.curselection()[0])
        self.relationHours.pop(self.usedArrays.curselection()[0])
        self.originalIndex.pop(self.usedArrays.curselection()[0])
        self.usedArrays.delete(self.usedArrays.curselection()[0])
    
        self.uvDisFig = plt.figure(2)
        plt.clf()
        for i in np.arange(len(self.arrays)):
            self.arrays[i] = interferometer.timetrack(arrayBox.configurations[1][self.originalIndex[i]],arrayBox.configurations[2][self.originalIndex[i]],299792458/(float(params.freq.get())*10**6),float(params.dec.get())*np.pi/180,self.relationHours[i])
            self.replot()
        self.plotUVDis.grid(column=2,row=1,rowspan=10,sticky=(N,E,W,S))
        
    def plotUV(self):
        if (((self.lastFreq == float(params.freq.get())) and (self.lastSampleTime == float(params.sampleTime.get())) and (self.lastDec == float(params.dec.get()))) or (len(uvDisplay.arrays) == 0)):
            #Replot the uv track with the new array added to it
            uvDisplay.replot()
            uvDisplay.plotUVDis.grid(column=2,row=1,rowspan=10)
            
        else:        
            uvDisplay.uvDisFig = plt.figure(2)
            plt.clf()
            for i in np.arange(len(uvDisplay.arrays)):
                uvDisplay.arrays[i] = interferometer.timetrack(arrayBox.configurations[1][uvDisplay.originalIndex[i]],arrayBox.configurations[2][uvDisplay.originalIndex[i]],299792458/(float(params.freq.get())*10**6),float(params.dec.get())*np.pi/180,uvDisplay.relationHours[i])
            uvDisplay.replot()
            uvDisplay.plotUVDis.grid(column=2,row=1,rowspan=10)
        self.lastFreq = float(params.freq.get())
        self.lastSampleTime = float(params.sampleTime.get())
        self.lastDec = float(params.dec.get())
        
        plt.savefig('UVTRACK.png')
        uvFrame.addImage(PIL.Image.open('UVTRACK.png'))


# In[147]:

class LoadButton:
    
    def __init__(self, parent, text='unamed'):
        self.parent = parent
        self.text = text
        self.path = StringVar()
        self.pixScale = StringVar()
        self.im = PIL.Image.new('RGB', (1,1), 'white')
        
        self.frame = Frame(self.parent)
        
        self.loader = Button(self.frame,text=self.text,command = self.openFile)
        self.loader.grid(column=1,row=1)
        
        self.loaded = ttk.Entry(self.frame, width=10, textvariable = self.path)
        self.loaded.grid(column=2,row=1)
        
        ttk.Label(self.frame, text="Pixel Scale (arcsec):").grid(column=1,row=2)
        self.pixScale.set(0.5)
        self.scaleEntry = Entry(self.frame, width=5, textvariable = self.pixScale)
        self.scaleEntry.grid(column=2, row = 2)
        
    #Load image for use in the calculations
    def LoadingImage(self):
        self.im = PIL.Image.open(self.loaded.get())
        refFrame.addImage(self.im)
        
    #Choose the path to open the image from, then run LoadingImage() to actually open it
    def openFile(self):
        self.imageLoaded = filedialog.askopenfilename()
        self.path.set(self.imageLoaded)
        self.LoadingImage()


# In[148]:

class RunImages:
    
    def __init__(self,parent,text='unamed'):
        self.parent = parent
        self.text = text
        
        self.frame = Frame(self.parent)
        
        self.run = Button(self.frame,text=self.text,command = self.FindEverything)
        self.run.grid(column=11,row=1)
     
    def FFTDisplay(self,image):
        fftim = np.fft.fft2(image)
        step = np.fft.fftshift(fftim)
        absoluteFFT = abs(step)
        logFFT = np.log(absoluteFFT+1)
        maxes = np.zeros(len(logFFT))
        for i in np.arange(len(logFFT)):
            maxes[i] = max(logFFT[i])
        totMax = max(maxes)
        moddedFFT = logFFT*255/totMax
        return PIL.Image.fromarray(moddedFFT)
    
    #This is the final calculation
    def FindEverything(self):
        #The Fourier Transform
        im = load.im.convert('L')
        imFourProto = self.FFTDisplay(im)
        fourFrame.addImage(imFourProto)
    
        #UV Track
        totalTrack = [[]]
        totalTrack.append([])
        for i in np.arange(len(uvDisplay.arrays)):
            for j in np.arange(len(uvDisplay.arrays[i][0])):
                totalTrack[0].append(uvDisplay.arrays[i][0,j])
                totalTrack[1].append(uvDisplay.arrays[i][1,j])
        totalTrackArr = np.array([totalTrack[0],totalTrack[1]])
        imCoverageProto = interferometer.makeMask(im,totalTrackArr,float(load.pixScale.get())).convert('1')
        #uvFrame.addImage(imCoverageProto)
    
        #Effective Beam
        imAndTrack = interferometer.observedFFT(im,imCoverageProto)
        abImAndTrack = np.log(abs(imAndTrack)+1)
        maxes = np.zeros(len(abImAndTrack))
        for i in np.arange(len(abImAndTrack)):
            maxes[i] = max(abImAndTrack[i])
        totMax = max(maxes)
        scaleImAndTrack = abImAndTrack*255/totMax
        finImAndTrack = PIL.Image.fromarray(scaleImAndTrack)
        fftuvFrame.addImage(finImAndTrack)
    
        #Sythesised Beam
        synBeam = abs(np.fft.fftshift(np.fft.ifft2(imCoverageProto.convert('L'))))
        logBeam = np.log(synBeam+1)
        maxes = np.zeros(len(logBeam))
        for i in np.arange(len(logBeam)):
            maxes[i] = max(logBeam[i])
        totMax = max(maxes)
        moddedBeam = logBeam*255/totMax
        synBeamim = PIL.Image.fromarray(moddedBeam)
        synFrame.addImage(synBeamim)
    
        #And now, the moment you've all been waiting for
        subImage = np.fft.ifft2(np.fft.ifftshift(imAndTrack))
        fixUp = np.log(abs(subImage)+1)
        maxes = np.zeros(len(fixUp))
        for i in np.arange(len(fixUp)):
            maxes[i] = max(fixUp[i])
        totMax = max(maxes)
        realFixUp = fixUp*255/totMax
        finImage = PIL.Image.fromarray(realFixUp)
        finFrame.addImage(finImage)


# In[157]:

#Final program run, there's quite a lot here

#Initialize everything
root = Tk()
root.title("Well, let's see")

#set up the master frame
frameMaster = ttk.Frame(root, padding = '5 5 5 5')
frameMaster.grid(column=0, row=0, sticky=(N,W,E,S))
frameMaster.columnconfigure(0, weight=1)
frameMaster.rowconfigure(0, weight=1)
frameMaster.configure(width=100,height=100)

#set up the tabs for the windows to view the frames
noteBook = ttk.Notebook(frameMaster)
noteBook.enable_traversal()
noteBook.grid(column=1,row=1)

#set up frame to handle imput and parameters of the uv track
frame = ttk.Frame(noteBook, padding = '5 5 5 5')
frame.grid(column=0, row=0, sticky=(N,W,E,S))
frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)
frame.configure(width=100,height=100)

#set up some initial arrays
usedArray = []

rB = np.array([0.4364, 1.4337, 2.8747, 4.7095, 6.9065, 9.4434, 12.3027, 15.4706, 18.9357,
          0.4840, 1.5899, 3.1881, 5.2229, 7.6595, 10.4728, 13.6438, 17.157, 21.,
          0.484, 1.5899, 3.1881, 5.2229, 7.6595, 10.4728, 13.6439, 17.1572, 21.])*10**3 #set all the r values in m
thetaA = np.zeros(len(rB)) #set all the theta values in degrees, 0 is parrallel to the surface, 90 points to the zennith
phiA = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
           125.0, 125.0, 125.0, 125.0, 125.0, 125.0, 125.0, 125.0, 125.0,
           245.0, 245.0, 245.0, 245.0, 245.0, 245.0, 245.0, 245.0, 245.0]) #set all the phi values in degrees, 0 is north, 90 is west
phiB = phiA*np.pi/180.
thetaB = thetaA*np.pi/180.
VLAcoords = np.array([rB,thetaB,phiB])
ASKAPcoords = np.array([[1000,1000,1000,1000,1000,1000,1000,1000],[0,0,0,0,0,0,0,0],[0,45/180*np.pi,90/180*np.pi,135/180*np.pi,180/180*np.pi,225/180*np.pi,270/180*np.pi,315/180*np.pi]])
ATCAcoords = np.array([[1000,2000,3000,4000,5000,6000],[0,0,0,0,0,0],[0,0,0,0,0,0]])

latitude = [34.0784*np.pi/180,34.0784*np.pi/180,34.0784*np.pi/180]

#set the names for the arrays
telNames = []
telNames.append('VLA')
telNames.append('ASKAP')
telNames.append('ATCA')

#arranges the array configuration coordinates into a lite
arrCoords = []
arrCoords.append(VLAcoords)
arrCoords.append(ASKAPcoords)
arrCoords.append(ATCAcoords)

arrayBox = ArraySelecter(frame, displayHeight=4, displayWidth=4,configurationsPOLAR=[telNames,arrCoords, latitude])
arrayBox.frame.grid(column=1,row=2, rowspan=5)

#button to add the selected array at the desired hour angle
add = AddButton(frame, text = "Add Selected")
add.frame.grid(column=2,row=5)

#Parameter setter
params = Parameters(frame, frequency=1428.5714285714284, sampleTime=60, declination=20)
params.frame.grid(column=2,row=2)

#hour slider canvas
hourSlide = HourSlider(frame,minHour=-12,maxHour=12, length = 250)
hourSlide.frame.grid(column=2,row=3)

#display the selected configurations for the calculations
uvDisplay = SelectedDisplay(frame, displayHeight=4, displayWidth=4, configurations=[])
uvDisplay.frame.grid(column=3,row=2,rowspan=5)

#Display Half
maneuver = 0

#set up frame to display results
imageFrame = ttk.Frame(noteBook)
imageFrame.grid(column=1,row=1,columnspan=20)

#Operations on the images parameters
load = LoadButton(imageFrame, text = 'Load File')
load.frame.grid(column=1, row = 1)

calculate = RunImages(imageFrame, text='Run')
calculate.frame.grid(column=11,row=1)

#Display reference image
refFrame = ImageFrame(imageFrame, width=250, height=250, text="Reference Image", relief=SUNKEN)
refFrame.frame.grid(column=1, row=5, columnspan=2, sticky=W)

#Display fft of reference image
fourFrame = ImageFrame(imageFrame, width=250, height=250, text = "Model FFT", relief=SUNKEN)
fourFrame.frame.grid(column=1, row=7, columnspan=2, sticky=W)

#Display uv track of reference image
uvFrame = ImageFrame(imageFrame, width=250, height=250, text = "UV Coverage", relief=SUNKEN)
uvFrame.frame.grid(column=4, row=5, sticky=W)

#Display fft with uv mask imposed
fftuvFrame = ImageFrame(imageFrame, width=250, height=250, text = "Observed FFT", relief=SUNKEN)
fftuvFrame.frame.grid(column=4, row=7, sticky=W)

#Display synthesiesed beam
synFrame = ImageFrame(imageFrame, width=250, height=250, text = "Synthesised Beam", relief=SUNKEN)
synFrame.frame.grid(column=7, row=5, sticky=W)

#Display ovserved final image
finFrame = ImageFrame(imageFrame, width=250, height=250, text = "Ovserved Image", relief=SUNKEN)
finFrame.frame.grid(column=7, row=7, sticky=W)

#Arrows (just aesthetics)
c1 = Canvas(imageFrame, width = 75, height = 30)
c1.create_line(50,0,50,30,arrow = LAST)
c1.grid(column=1,row=6, sticky = E)

c2 = Canvas(imageFrame, width = 75, height = 30)
c2.create_line(50,0,50,30,arrow = LAST)
c2.grid(column=4,row=6)

c3 = Canvas(imageFrame, width = 75, height = 75)
c3.create_line(0,50,75,50,arrow = LAST)
c3.grid(column=3,row=7)

c4 = Canvas(imageFrame, width = 75, height = 75)
c4.create_line(0,50,75,50,arrow = LAST)
c4.grid(column=6,row=5)

c5 = Canvas(imageFrame, width = 75, height = 75)
c5.create_line(0,50,75,50,arrow = LAST)
c5.grid(column=6,row=7)

for child in frame.winfo_children():
    child.grid_configure(padx=5, pady=5)

#Set up tabs in the overall notebook
noteBook.add(frame, text='Set UV Track', underline=0, padding=2)
noteBook.add(imageFrame, text='Output', underline=0, padding=2)

root.mainloop()

