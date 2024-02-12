import customtkinter as ctk
import tkinter as tk
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
from tkinter import messagebox
from PIL import ImageTk,Image


# Load the weights in YOLO model
model=YOLO(r'YOLO WEIGHTS/yolov8s.pt')


# Creates a list containing the different categories that can be detected via YOLO app
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")


# Sets the appearance of the window
# Supported modes : Light, Dark, System
ctk.set_appearance_mode("Dark") 


# Sets the color of the widgets in the window
# Supported themes : green, dark-blue, blue 
ctk.set_default_color_theme("green") 


# App Class
class App(ctk.CTk):
	# The layout of the window will be written
	# in the init function itself
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


		# Sets the title of the window 
		self.title("YOLO App") 

        
		# Sets the dimensions of the window 
		self.attributes('-fullscreen',True)


		# Screen Resolution
		self.screen_width = self.winfo_screenmmwidth()
		self.screen_height = self.winfo_screenheight()


        # Intialize the filepath to NONE
		self.file_path = None


        # Title Label
		self.nameLabel = ctk.CTkLabel(self,
								text="YOLO APPLICATION",font=("Arial", 70, "bold"),text_color='cyan')
		self.nameLabel.place(relx=0.5,rely=0.09,anchor='center')


		# Type of Function Label
		self.typeOFFunction = ctk.CTkLabel(self,
									text="Function want to Perform",font=("Arial",25,'bold'))

		self.typeOFFunction.place(relx=0.05,rely=0.20)


		# Type of  combo box
		self.functionOptionMenu = ctk.CTkOptionMenu(self,
									values=["Normal Object Detection",
									"Count Number of Objects"])
		self.functionOptionMenu.configure(width=180)

		self.functionOptionMenu.place(relx=0.30,rely=0.2)


    	# Object Type Label
		self.objectType = ctk.CTkLabel(self,
									text="Object to Detect",font=("Arial",25,'bold'))
		self.objectType.place(relx=0.05,rely=0.35)


		# Object Type combo box
		values=class_list.copy()
		values.insert(0,'Default')
		self.objectTypeOptionMenu=ctk.CTkOptionMenu(self,values=values)
		self.objectTypeOptionMenu.place(relx=0.30,rely=0.35)
		self.objectTypeOptionMenu.configure(width=180)


        # Type of input label
		self.inputType = ctk.CTkLabel(self,
									text="Input Type",font=("Arial",25,'bold'))

		self.inputType.place(relx=0.05,rely=0.50)


		# Type of  input radio button
		self.inputVar = tk.StringVar(value="Image")
		self.imageRadioButton = ctk.CTkRadioButton(self,
                                   text="Image",font=("Arial",15,'bold'),
                                   variable=self.inputVar,
                                   value="image",command=self.enable_disable,border_color='black')
		self.imageRadioButton.place(relx=0.27,rely=0.50)
		self.videoRadioButton = ctk.CTkRadioButton(self,
                                     text="Video",font=("Arial",15,'bold'),
                                     variable=self.inputVar,
                                     value="video",command=self.enable_disable,border_color='black')
		self.videoRadioButton.place(relx=0.345,rely=0.50)

		
        # Browse Button 
		self.browse = ctk.CTkButton(self,
							  text='Choose File',
							  command=self.select_file,
							  )
		self.browse.place(relx=0.20,rely=0.58)
		self.browse.configure(state='disabled')


		# Generate Button
		self.generateResultsButton = ctk.CTkButton(self,
										text="Generate Results",font=('Arial',15,'bold'),
										command=self.YOLO_Inferrence)
		self.generateResultsButton.place(relx=0.20,rely=0.67)


        # Close Button
		self.quitButton = ctk.CTkButton(self,
										text="CLOSE",fg_color='white',text_color='black',hover_color='red',
										command=self.destroy)
		self.quitButton.place(relx=0.50,rely=0.95,anchor='center')


		# Create a Canvas for displaying results
		self.canvas = ctk.CTkCanvas(self,
							    width=640, height = 360,
								highlightbackground= 'green', highlightcolor= 'green',
								bg='grey',
								highlightthickness=5
								)
		self.canvas.place(relx=0.48,rely=0.18)

		
		# Setting the background image for canvas
		self.img = cv2.cvtColor(cv2.resize(cv2.imread(r"data/canvas background.webp"),(640,360)),cv2.COLOR_BGR2RGB)
		self.img = ImageTk.PhotoImage(Image.fromarray(self.img))
		self.canvasBackground=self.canvas.create_image(5, 5, image=self.img ,anchor='nw')


		# TextBox for displaying the results if 'Count Number of Objects' is selected
		self.textBox = ctk.CTkTextbox(self,width=650,height=100,font=('Arial',20,'bold'))

		
	# This function is used to browse the files from system
	def select_file(self):

		radio_button=self.inputVar.get()

		if radio_button=='image':
			self.file_path = ctk.filedialog.askopenfilename(filetypes=[("jpeg files", "*.jpg;*.png;*.webp;*.jpeg")])

		elif radio_button=='video':
			self.file_path = ctk.filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.flv;*.avi;*.mkv")])

			
    # This function is used to enable or disable the browse button
	def enable_disable(self):

		if self.inputVar.get() == 'image' or self.inputVar.get() == 'video':
			self.browse.configure(state='normal')

		else:
			self.browse.configure(state='disabled')

	
	# This function is used for performing detection 
	def detection(self,source,coco_names,function_type):

		object_type = self.objectTypeOptionMenu.get()
		results=model.predict(source,verbose=False)
		a=results[0].boxes.data
		px=pd.DataFrame(a).astype("float")
		self.objectList=[]

		for index,row in px.iterrows():
			x1=int(row[0])
			y1=int(row[1])
			x2=int(row[2])
			y2=int(row[3])
			d=int(row[5])
			c=class_list[d]
			self.objectList.append([x1,y1,x2,y2,c])
			
		bbox_id=Tracker().update(self.objectList)
		i=0
		for bbox in bbox_id:
			x3,y3,x4,y4,c,id=bbox
			if object_type=='Default':
				if function_type=="Normal Object Detection":	
					source = cv2.rectangle(source,(x3,y3),(x4,y4),(0,0,255),2)

					if i%2==0:
						cv2.putText(source, c, (x3, y3+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1)

					else:
						cv2.putText(source, c, (x3, y4-2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1)

				elif function_type=="Count Number of Objects":
					if c in coco_names.keys():
						if id not in self.objectid:
							self.objectid.append(id)
							coco_names[c] = coco_names[c]+1
			else:
				if object_type in c:
					if function_type=="Normal Object Detection":
						source = cv2.rectangle(source,(x3,y3),(x4,y4),(0,0,255),2)

						if i%2==0:
							cv2.putText(source, c, (x3, y3+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1)

						else:
							cv2.putText(source, c, (x3, y4-2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1)

					elif function_type=="Count Number of Objects":
						if id not in self.objectid:	
							self.objectid.append(id)
							coco_names[c] = coco_names[c]+1
				else:
					continue

		if function_type=="Normal Object Detection":
			result=source.copy()

		elif function_type=="Count Number of Objects":
			for key,value in coco_names.items():
				if value>0:
					self.detectionResults[key]=value

			result = self.detectionResults.copy()
		
		return result			


	# This function shows the result of 'Count Number of Objects' in textBox
	def create_text(self,result):
		text=""
		text = f"Total Number of Objects Detected : {len(self.objectid)}\n"
		for key,value in result.items():
			text+=f"{key} : {value}   "

		return text
		


    # This function is used to perform inference on desired input
	def YOLO_Inferrence(self):
		
		self.coco_names = {}
		for name in class_list:
			self.coco_names[name]=0
		self.textBox.delete("0.0", "end")
		self.textBox.pack_forget()
		self.detectionResults={}
		self.objectid=[]
		self.function_type = self.functionOptionMenu.get()

		# For image processing
		if self.inputVar.get()=='image':
			if self.file_path=='' or self.file_path==None:
				messagebox.showwarning("WARNING","Please Select a Image First")

			else:
				img=cv2.imread(self.file_path)
				result=self.detection(img,self.coco_names,self.function_type)
				if len(result)!=0:
					if self.function_type=="Normal Object Detection":
						result = cv2.resize(result,(640,360))
						result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)				
						self.img = ImageTk.PhotoImage(image = Image.fromarray(result))
						self.canvas.itemconfigure(self.canvasBackground, image=self.img)

					elif self.function_type=="Count Number of Objects":
						img = cv2.resize(img,(640,360))
						img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)				
						self.img = ImageTk.PhotoImage(image = Image.fromarray(img))
						self.canvas.itemconfigure(self.canvasBackground, image=self.img)
						self.textBox.pack(anchor='ne',side='bottom',padx=60,pady=144)
						self.textBox.delete("0.0", "end")
						text = self.create_text(result)
						self.textBox.insert("0.0",text)

				else:
					self.textBox.delete("0.0", "end")
					self.textBox.insert("0.0","No Object Detected!")

		# For video processing
		elif self.inputVar.get()=='video':
			if self.file_path=='' or self.file_path==None:
				messagebox.showwarning("WARNING","Please Select a Video First")
				
			self.cap=cv2.VideoCapture(self.file_path)

			while True:
				ret,frame = self.cap.read()
				x = int(self.screen_width*1.81)
				y = int(self.screen_height*0.145)
				cv2.namedWindow("video")
				cv2.moveWindow("video", x, y)

				if not ret:
					break

				result=self.detection(frame,self.coco_names,self.function_type)
				if len(result)!=0:
					if self.function_type=="Normal Object Detection":
						result = cv2.resize(result,(640,360))
						cv2.imshow("video",result)
						self.img = cv2.cvtColor(cv2.resize(cv2.imread(r"data/canvas background.webp"),(640,360)),cv2.COLOR_BGR2RGB)
						self.img = ImageTk.PhotoImage(Image.fromarray(self.img))
						self.canvas.itemconfigure(self.canvasBackground, image=self.img)		
					

					elif self.function_type=="Count Number of Objects":
						frame = cv2.resize(frame,(640,360))
						cv2.imshow("video",frame)
						self.img = cv2.cvtColor(cv2.resize(cv2.imread(r"data/canvas background.webp"),(640,360)),cv2.COLOR_BGR2RGB)
						self.img = ImageTk.PhotoImage(Image.fromarray(self.img))
						self.canvas.itemconfigure(self.canvasBackground, image=self.img)
						self.textBox.pack(anchor='ne',side='bottom',padx=60,pady=144)
						self.textBox.delete("0.0", "end")
						text = self.create_text(result)
						self.textBox.insert("0.0",text)

				else:
					self.textBox.delete("0.0", "end")
					self.textBox.insert("0.0","No Object Detected!")	

				if cv2.waitKey(1)==27:
					break

			self.cap.release()	
			cv2.destroyAllWindows()

		else:
			messagebox.showwarning("WARNING","Please Choose a Input Type")

		# Change file path to none after task is completed
		self.file_path=None



if __name__ == "__main__":

	app = App()

	# Used to run the application
	app.mainloop()	 
