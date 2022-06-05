from window.Window import Window
from os.path import isfile

if __name__ == "__main__":
    # Check if trained model is provided
    if not isfile('model.pth'):
        print('\"model.pth\" not found: Please provide a trained model!')
    else:
        print(f"Starting application...")
        # Start the GUI
        Window()
