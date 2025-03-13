`original_py.py` contains an extract of some not-great code for driving a popular kind of microcontroller.
For simplicity we've stubbed out all the necessary library functions in `libs_py.py`.

We'd like you to refactor the code into a form that's more maintainable.
Use (or don't use) any parts of the language and standard library you think are appropriate, and change as much of the structure and naming of the code as you think is right.
For this exercise, tiny behavioural modifications that don't have significant user-visible effects still count as refactoring.
As far as efficiency goes, you can assume the code is running on standard desktop-class hardware now.

The program assumes you have an LCD wired into the microcontroller, which you control with the LiquidCrystal class.
This class is initialised with the numbers of the relevant pins on the chip.
You can safely treat this numbering as opaque and just use the provided values.

It also assumes a set of buttons is connected on pin 0, which the user can press to change which menu item is selected.

The toolchain requires you to define two functions: `setup()` which is called once at init, and `loop()` which is called again and again forever after setup.
Since calling these is normally handled for you by the platform, we've added a `main()` function to do the same thing.
Please keep that structure.

You can write your code in `your_version_py.py`.

Do get in touch with us if there's any confusion!
Email `jthompson@symmetryinvestments.com`, starting the subject line with "CODE TEST ADVICE" to make sure we see it quickly.
