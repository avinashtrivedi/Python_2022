##  ----------------------------------------------------------------------
##  LCD driver

## All methods here are stubs, replacing real library functionality it's too
## annoying to set up a toolchain for.
##
class LiquidCrystal:
    def __init__(self, pinNumRS, pinNumEnable, pinNumD4, pinNumD5,
                       pinNumD6, pinNumD7):
        """Sets up the connection between the screen and the microcontroller"""
        pass

    def clear(self):
        """Clears the LCD screen and positions the cursor in the upper-left corner"""
        pass

    def begin(self, cols, rows):
        """Initializes the interface to the LCD screen, and specifies the dimensions
        (width and height) of the display.
        begin() needs to be called before any other LCD library commands."""
        pass

    def setCursor(self, col, row):
        """Position the LCD cursor; that is, set the location at which subsequent
        text written to the LCD will be displayed."""
        pass

    def print(self, text):
        """Prints text to the LCD."""
        pass
    
    def write(self, byte):
        """Write a character to the LCD."""
        pass

    def createChar(self, num, data):
        """Create a custom character (glyph) for use on the LCD."""
        print("createChar")
        pass


##  ----------------------------------------------------------------------
##  Maths (ish)

def constrain(val, low, high):
    if val < low:
        return low
    elif val < high:
        return val
    else:
        return high


##  ----------------------------------------------------------------------
##  Hardware interaction

def analogRead(pinNum):
    """Reads voltage off a microcontroller pin.
    Again, stub function."""
    return -1
