
# coding: utf-8
from tkinter import *
from tkinter import ttk
import tkinter as tk
import PIL
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from pylab import rcParams
get_ipython().magic('matplotlib inline')

class ArrayCreater:
    def __init__(self,parent, pos):
        self.pos = pos
        
        self.frame = ttk.Frame(parent, padding = '5 5 5 5')
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        self.frame.configure(width=500,height=500)
        ttk.Label(self.frame,text = '----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------').grid(column=0,row=0, columnspan = 100)

        self.canvasSizeMin = StringVar()
        self.canvasSizeMin.set(0)
        self.canvasSizeMax = StringVar()
        self.canvasSizeMax.set(1000)

        rcParams['figure.figsize'] = 3, 3
        self.arr = plt.figure(1)
        self.u = self.pos[:,0]
        self.v = self.pos[:,1]
        self.xLim = np.array([float(self.canvasSizeMin.get()),float(self.canvasSizeMax.get())])
        self.yLim = np.array([float(self.canvasSizeMin.get()),float(self.canvasSizeMax.get())])
        plt.xlim(self.xLim[0],self.xLim[1])
        plt.ylim(self.yLim[0],self.yLim[1])
        plt.plot(self.u,self.v,'.')

        self.maps = FigureCanvasTkAgg(self.arr, master=self.frame)
        self.plotWidget = self.maps.get_tk_widget()

        self.canvis = self.plotWidget.grid(column=1, row=1, rowspan=10)

        self.uCoord = []
        self.vCoord = []
        self.aNum = StringVar()
        self.aNum.set(len(self.pos))

        ttk.Label(self.frame, text = 'Number of Antenna:').grid(column=4,row=3,sticky=W)
        aEntry = ttk.Entry(self.frame, width=15, textvariable=self.aNum).grid(column=4,row=4,sticky=W)
        ttk.Button(self.frame, text="Set", command=self.setNum).grid(column=4, row=5, sticky=W)
        ttk.Label(self.frame, text = 'Canvas Size').grid(column=4,row=1,sticky=W)
        ttk.Entry(self.frame, textvariable = self.canvasSizeMin).grid(column=4,row=2,sticky=W)
        ttk.Entry(self.frame, textvariable = self.canvasSizeMax).grid(column=5,row=2,sticky=W)

        self.uEntry = []
        self.vEntry = []

        self.scrollFrame = ttk.Frame(self.frame)
        self.scrollCanvas= Canvas(self.scrollFrame)
        self.scrollbox = ttk.Frame(self.scrollCanvas)
        self.scrollbox.grid(column=0,row=0,sticky =(N,S))
        self.scroller=Scrollbar(self.scrollFrame,orient="vertical",command=self.scrollCanvas.yview)
        self.scrollCanvas.configure(yscrollcommand=self.scroller.set)

        self.scroller.grid(column=1,row=0, rowspan=10, sticky = (N,S))
        self.scrollFrame.grid(column=3,row=6)
        self.scrollCanvas.grid(column=0,row=0)
        self.scrollCanvas.create_window((0,0),window=self.scrollbox)
        self.frame.bind("<Configure>",self.scroll)

        ttk.Label(self.scrollbox, text = 'u coordinate').grid(column=0,row=0,sticky=W)
        ttk.Label(self.scrollbox, text = 'v coordinate').grid(column=1,row=0,sticky=W)
        for i in np.arange(int(self.aNum.get())):
            self.uCoord.append(StringVar())
            self.uCoord[i].set(self.pos[i,0])
            self.uEntry.append(ttk.Entry(self.scrollbox, width=10, textvariable=self.uCoord[i]))
            self.uEntry[i].grid(column=0,row = i+1, sticky=W)
            self.vCoord.append(StringVar())
            self.vCoord[i].set(self.pos[i,1])
            self.vEntry.append(ttk.Entry(self.scrollbox, width=10, textvariable=self.vCoord[i]))
            self.vEntry[i].grid(column=1,row = i+1, sticky=W)

        self.goButton = ttk.Button(self.frame, text="Go", command=self.reposition)
        self.goButton.grid(column=4, row=5+int(self.aNum.get()), sticky=W)

        self.mapLabel = ttk.Label(self.frame, text='0')
        self.mapLabel.grid(column=1,row=11)
        self.mapLabel.configure(width=100)

        self.pointX = StringVar()
        self.pointY = StringVar()

        self.mapX = ttk.Label(self.frame, textvariable=self.pointX)
        self.mapX.grid(column=1,row=102)
        self.mapX.configure(width=100)

        self.mapY = ttk.Label(self.frame, textvariable = self.pointY)
        self.mapY.grid(column=1,row=103)
        self.mapY.configure(width=100)

        self.plotWidget.bind('<Enter>', lambda e: self.mapLabel.configure(text='1'))
        self.plotWidget.bind('<Leave>', lambda e: self.mapLabel.configure(text='0'))
        self.plotWidget.bind('<Motion>', lambda e: self.mouse(e))
        self.plotWidget.bind('<Button - 1>',lambda e: self.addPoint(e))
        self.plotWidget.bind('<Button - 3>',lambda e: self.removePoint(e))
        
    def reposition(self):
        try:
            for i in np.arange(len(self.pos)):
                uVal = float(self.uCoord[i].get())
                vVal = float(self.vCoord[i].get())
                self.pos[i] = np.array([uVal,vVal])
            self.u = self.pos[:,0]
            self.v = self.pos[:,1]
            self.xLim = np.array([float(self.canvasSizeMin.get()),float(self.canvasSizeMax.get())])
            self.yLim = np.array([float(self.canvasSizeMin.get()),float(self.canvasSizeMax.get())])
            plt.clf()
            plt.xlim(self.xLim[0],self.xLim[1])
            plt.ylim(self.yLim[0],self.yLim[1])
            plt.plot(self.u,self.v,'.')
            self.maps = FigureCanvasTkAgg(self.arr, master=self.frame)
            self.plotWidget = self.maps.get_tk_widget()
            self.canvis = self.plotWidget.grid(column=1, row=1, rowspan=10)
        
            self.mapLabel = ttk.Label(self.frame, text='0')
            self.mapLabel.grid(column=1,row=11)
            self.mapLabel.configure(width=100)
        
            self.mapX = ttk.Label(self.frame, text='0')
            self.mapX.grid(column=1,row=12)
            self.mapX.configure(width=100)

            self.mapY = ttk.Label(self.frame, text='0')
            self.mapY.grid(column=1,row=13)
            self.mapY.configure(width=100)

            self.plotWidget.bind('<Enter>', lambda e: self.mapLabel.configure(text='1'))
            self.plotWidget.bind('<Leave>', lambda e: self.mapLabel.configure(text='0'))
            self.plotWidget.bind('<Motion>', lambda e: self.mouse(e))
            self.plotWidget.bind('<Button - 1>',lambda e: self.addPoint(e))
            self.plotWidget.bind('<Button - 3>',lambda e: self.removePoint(e))
        except ValueError:
            print('error')
            
    def setNum(self):
        try:
            self.goButton.grid_forget()
            for i in np.arange(len(self.pos)):
                self.uEntry[i].grid_forget()
                self.vEntry[i].grid_forget()
            g = []
            while (len(g) < int(self.aNum.get())):
                if (len(g) < len(self.pos)):
                    g.append(self.pos[len(g)])
                else:
                    g.append(np.array([0,0]))
            self.pos = np.array([])
            self.pos = np.asarray(g)
        
            self.scrollFrame.grid_forget()
            self.scrollCanvas.grid_forget()
            self.scrollbox.grid_forget()
            self.scroller.grid_forget()
        
            self.uEntry = []
            self.vEntry = []
        
            self.scrollFrame = ttk.Frame(self.frame)
            self.scrollCanvas= Canvas(self.scrollFrame)
            self.scrollbox = ttk.Frame(self.scrollCanvas)
            self.scrollbox.grid(column=0,row=0,sticky =(N,S))
            self.scroller=Scrollbar(self.scrollFrame,orient="vertical",command=self.scrollCanvas.yview)
            self.scrollCanvas.configure(yscrollcommand=self.scroller.set)

            self.scroller.grid(column=1,row=0, rowspan=10, sticky = (N,S))
            self.scrollFrame.grid(column=3,row=6)
            self.scrollCanvas.grid(column=0,row=0)
            self.scrollCanvas.create_window((0,0),window=self.scrollbox)
            self.frame.bind("<Configure>",self.scroll)

            ttk.Label(self.scrollbox, text = 'u coordinate').grid(column=0,row=0,sticky=W)
            ttk.Label(self.scrollbox, text = 'v coordinate').grid(column=1,row=0,sticky=W)
            for i in np.arange(int(self.aNum.get())):
                self.uCoord.append(StringVar())
                self.uCoord[i].set(self.pos[i,0])
                self.uEntry.append(ttk.Entry(self.scrollbox, width=10, textvariable=self.uCoord[i]))
                self.uEntry[i].grid(column=0,row = i+1, sticky=W)
                self.vCoord.append(StringVar())
                self.vCoord[i].set(self.pos[i,1])
                self.vEntry.append(ttk.Entry(self.scrollbox, width=10, textvariable=self.vCoord[i]))
                self.vEntry[i].grid(column=1,row = i+1, sticky=W)
            
            self.goButton = ttk.Button(self.frame, text="Go", command=self.reposition)
            self.goButton.grid(column=4, row=5+int(self.aNum.get()), sticky=W)
        except ValueError:
            print('error')
    
    def addPoint(self,e):
        try:
            self.aNum.set(int(self.aNum.get())+1)
            self.setNum()
            
            self.uCoord[int(self.aNum.get())-1].set(self.pointX.get())
            self.vCoord[int(self.aNum.get())-1].set(self.pointY.get())
            self.reposition()
        except ValueError:
            print('error')
            
    def removePoint(self,e):
        try:
            i = 0
            while i < len(self.pos):
                if ((float(self.uCoord[i].get()) < float(self.pointX.get())+20) and (float(self.uCoord[i].get()) > float(self.pointX.get())-20) and (float(self.vCoord[i].get()) < float(self.pointY.get())+20) and (float(self.vCoord[i].get()) > float(self.pointY.get())-20)):
                    for j in np.arange(len(self.pos)-i-1):
                        self.pos[i+j] = self.pos[i+j+1]
                    self.aNum.set(int(self.aNum.get())-1)
                i += 1
            self.setNum()
            self.reposition()
        except ValueError:
            print('Error')
            
    def mouse(self,e):
        size = 300
        start = size*np.array([19/150,3/25])
        end = size-size*np.array([1/10,19/150])
        step = np.array([(float(self.canvasSizeMax.get())-float(self.canvasSizeMin.get())+2)/(end[0]-start[0]),(float(self.canvasSizeMax.get())-float(self.canvasSizeMin.get())+2)/(end[1]-start[1])])
        self.pointX.set(((e.x-start[0])*step[0]+float(self.canvasSizeMin.get())-1))
        self.pointY.set((-((e.y-start[1]))*step[1]+float(self.canvasSizeMax.get())+1))
        
    def scroll(self,event):
        self.scrollCanvas.configure(scrollregion=self.scrollCanvas.bbox("all"),width=200,height=200)



