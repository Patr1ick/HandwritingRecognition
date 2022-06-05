import torch
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw
from model import convertImageToTensor, saveImage
from model.NeuralNetwork import NeuralNetwork


class Window():
    """
    Class for the GUI

    Attributes
    ----------
    canvas_width: int
        The width of the canvas (default is 400)
    canvas_height: int
        The height of the canvas (default is 400)
    root: Tk
        The root window
    canvas: Canvas
        The canvas on which one draws
    label: Lable
        The label to display the results
    is_drawing: bool
        This bool indicates wether the user is drawing or not (default is False)
    img: Image
        The image in the background which will be saved and converted to the input
    draw: ImageDraw.Draw
        Interface to draw on PIL Images
    classes: List
        A list with all classes that can be recognized
    model: NeuralNetwork
        The object of the neural network

    Methods
    -------
    clear:
        Clears the canvas and the image.
    onMouseClick(event)
        Event when the user clicks on the canvas
    onMouseRealease(event):
        Event when the user realese the mouse on the canvas
    isDrawing(event)
        Event when the user is moving its mouse and drawing
    onButtonClick()
        When the user click on the recognize button
    preview()
        Shows the preview via Matplotlib
    createResult(input:torch.Tensor)
        Compute the result from the input and show it 
    """

    def __init__(self) -> None:
        # Values
        self.canvas_width = 400
        self.canvas_height = 400

        # Main Window
        self.main = Tk()
        self.main.title('Draw a character or number to recognize')
        self.main.resizable(False, False)
        self.main.configure(background='white')

        # Groups to arrange the nodes
        groupDraw = LabelFrame(self.main, text="Draw", background="#FFF", )
        groupDraw.pack(side='left', padx=10, pady=10)

        groupControl = LabelFrame(
            groupDraw, text="Controls", background="#FFF")
        groupControl.pack(side='bottom', pady=10)

        groupResult = LabelFrame(
            self.main, text="Result", background="#FFF", width=300)
        groupResult.pack(side='right', padx=10, pady=10)

        # Canvas
        self.canvas = Canvas(groupDraw, width=self.canvas_width,
                             height=self.canvas_height)
        self.canvas.pack(side='top', expand=NO, fill=BOTH, padx=20, pady=20)
        self.canvas.bind('<ButtonPress-1>', self.onMouseClick)
        self.canvas.bind('<ButtonRelease-1>', self.onMouseRelease)
        self.canvas.bind('<Motion>', self.isDrawing)

        # Buttons
        button = Button(groupControl, text="Compute",
                        command=self.onButtonClick)
        button.pack(side='right', padx=5, pady=10)

        button = Button(groupControl, text="Preview", command=self.preview)
        button.pack(side='right', padx=5, pady=10)

        button = Button(groupControl, text="Clear", command=self.clear)
        button.pack(side='right', padx=5, pady=10)

        # Result label
        self.label = Label(
            groupResult, text="No results.", anchor='nw', wraplength=300, justify="left", font=("Courier", 16))
        self.label.pack(side='top', padx=10, pady=10)

        # Background logic
        self.isDrawing = False
        self.img = Image.new(
            'RGB', (self.canvas_width, self.canvas_height), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.img)

        # Neural network
        self.classes = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
        ]

        self.model = NeuralNetwork(len(self.classes))
        # Load pre-trained network
        self.model.load_state_dict(torch.load('model.pth'))
        # Set the model to the evaluation mode
        self.model.eval()

        self.main.mainloop()

    def onMouseClick(self, event):
        """Event method for the canvas if the mouse is pressed

        TestCase
        --------
        If the mouse is pressed the bool is set to True
        """
        self.isDrawing = True

    def onMouseRelease(self, event):
        """Event method for the canvas if the mouse is released

        TestCase
        --------
        If the mouse is released the bool is set to False
        """
        self.isDrawing = False

    def isDrawing(self, event):
        """Event function for the canvas

        TestCase
        --------
        If the mouse is pressed aroung the cursor a ellipse/circle should be drawn.
        """
        if self.isDrawing:
            # Calculate the x, y coordinates for the drawing
            x1, y1 = (event.x - 16), (event.y - 16)
            x2, y2 = (event.x + 16), (event.y + 16)
            # Draw it on the canvas
            self.canvas.create_oval(x1, y1, x2, y2, fill='#000000')
            # Draw it on the PIL Image
            self.draw.ellipse([x1, y1, x2, y2], fill='white', outline='white')

    def clear(self):
        """Clear the canvas, PIL Image and the label

        TestCase
        --------
        Run the method and check if the canvas is cleared, every pixel of the PIL Image black and the text of the label empty
        """
        self.canvas.delete('all')
        self.draw.rectangle((0, 0, 400, 400), fill='black')
        self.label['text'] = ""

    def onButtonClick(self) -> None:
        """The method if the "Recognize" button is pressed

        TestCase
        --------
        If the button pressed some kind of result should be created and shown to the label. Additionally the tensor from the image could be checked.

        """
        saveImage(self.img)
        tensor = convertImageToTensor()
        self.createResult(tensor)

    def preview(self) -> None:
        """Preview how it is passed to the neural network

        TestCase
        --------
        If the button "Preview" is pressed a matplotlib window should open.
        """
        if self.canvas.find_all() == ():
            print("You have to draw something before you can preview it.")
            return
        saveImage(self.img)
        tensor = convertImageToTensor()
        plt.imshow(tensor.reshape(28, 28), cmap="binary")
        plt.title(
            "Preview of the image how it will be passed to the neural network")
        plt.show()

    def createResult(self, input_tensor: torch.Tensor) -> None:
        """Method to create a result from an input tensor and update the label.

        Parameters
        ----------
        input_tensor: torch.Tensor
            The input data as a PyTorch Tensor

        TestCase
        --------
        If the method is called it can be checked if in the end the label text changed. As input pre-defined tensors can be passed an checked if the label shows the correct results or the same results which the neural net would return

        """
        # Convert the tensor to the correct input shape
        input_tensor = input_tensor.reshape(1, 1, 28, 28)
        # Fit it into the network and get the results
        with torch.no_grad():
            output = self.model(input_tensor)

        # Compute the probabilites over the predicted ouput classes with the softmax function
        output = torch.softmax(
            output, -1, torch.double).numpy().reshape(len(self.classes))
        # Get the best guess
        output_class = np.argmax(output)
        # Generate text for the labels
        text = f"Best Result: {self.classes[output_class]}\nProbabilites:"
        for i, c in enumerate(self.classes):
            value = round(output[i], 2)*100
            if value != 0.0:
                text += f"\n{c}: {value:.2f}%"

        self.label['text'] = text
        print(text)
