from libs_py import *


##  ----------------------------------------------------------------------
##  Screen handling

lcd = LiquidCrystal(8, 9, 4, 5, 6, 7)

menuPage = 0
menuItems = ["START CAPTURE", "START SHOWCASE" ,"PRESETS",
             "SET TRIGGER", "SETTINGS", "ABOUT"]
maxMenuPages = round((len(menuItems) / 2) + 0.5)

## Bitmap icons drawn on screen
downArrow = [ 0b00100, 0b00100,
              0b00100, 0b00100,
              0b00100, 0b11111,
              0b01110, 0b00100 ]
upArrow   = [ 0b00100, 0b01110,
              0b11111, 0b00100,
              0b00100, 0b00100,
              0b00100, 0b00100 ]

def mainMenuDraw():
    lcd.clear()
    lcd.setCursor(1, 0)
    lcd.print(menuItems[menuPage])
    lcd.setCursor(1, 1)
    lcd.print(menuItems[menuPage + 1])
    if menuPage == 0:
        lcd.setCursor(15, 1)
        lcd.write(2)
    elif menuPage > 0 and menuPage < maxMenuPages:
        lcd.setCursor(15, 1)
        lcd.write(2)
        lcd.setCursor(15, 0)
        lcd.write(1)
    elif menuPage == maxMenuPages:
        lcd.setCursor(15, 0)
        lcd.write(1)
    

##  ----------------------------------------------------------------------
##  Reacting to button presses

def evaluateButton(x):
    """This function is called whenever a button press is evaluated.
    The LCD shield works by observing a voltage drop across the buttons all
    hooked up to A0."""
    result = 0
    if x < 50:
        result = 1  # up
    elif x < 195:
        result = 2  # down
    return result

readKey = 0

def operateMainMenu():
    activeButton = 0
    while activeButton == 0:
        button = 0
        global readKey
        global menuPage
        readKey = analogRead(0)
        button = evaluateButton(readKey)
        if button == 0:
            pass  # When button returns as 0 there is no action taken
        elif button == 1:
            button = 0
            menuPage = menuPage - 1
            menuPage = constrain(menuPage, 0, maxMenuPages)
            mainMenuDraw()
            activeButton = 1
        elif button == 2:
            button = 0
            menuPage = menuPage + 1
            menuPage = constrain(menuPage, 0, maxMenuPages)
            mainMenuDraw()
            activeButton = 1


##  ----------------------------------------------------------------------
##  Top level 

def setup():
    lcd.begin(16, 2)
    lcd.createChar(1, upArrow)
    lcd.createChar(2, downArrow)

def loop():
    mainMenuDraw()
    operateMainMenu()

# To simulate the rest of the toolchain
def main():
    setup()
    while True:
        loop()

main()
