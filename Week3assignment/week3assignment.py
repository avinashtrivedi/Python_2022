#For this assignment, you are going to pretend that you are writing an application that uses JSON to store configuration data in it.

#First, you are going to start by writing a JSON file called ("basic_config.json") that stores the following variables and their associated values:

#1. Safe Mode - "On"
#2. Memory - "32MB"
#3. Error Log - "logs/errors.log"

#Your program is going to load all of these values up on startup, display them just like I have them above, and then give the user the option to either modify an existing value or add new ones.

#Use your program to add the following configuration options (note that you are not limited to just these - I am asking you to add them using your program):

#Allow File Uploads - "Yes"
#Use Caching - "Yes"
#Caching File - "cache/filecache.cache"
#Mail Host - "mail.apphost.com"

#Also, give the user the option to either save their changes or discard them. If the user chooses to accept the changes, the new configuration will be saved as config_override.json. Under no circumstances should basic_config.json ever be overwritten.

#On subsequent program runs, in order to make this work properly, you are going to need to implement a check to see if config_override.json exists before loading the information in basic_config.json. In other words, once basic_config.json has been overridden, you should always load config_override.json from that point on to make your changes and adjustments. When you save back, you must have a means in place to backup your old file. You do not need to have functionality in place to have your applications load or restore data from old backups.

#Users may modify the value of any setting they wish, but they can only delete configurations for items not in the original configuration such as:

#Allow File Uploads - "Yes"
#Use Caching - "Yes"
#Caching File - "cache/filecache.cache"
#Mail Host - "mail.apphost.com"

#In other words, the keys contained in the basic_config.json file are all required. You must add logic and decide the best way to delete items the user wishes to remove.

#Finally, you will create functions to (minimally - you can and probably should write more functions than the ones listed below) handle the following tasks:

#Loading the information from the proper configuration file
#Saving configuration data.
#Adding a configuration
#Please note that you must organize your functions into modules and produce documentation (using Pydoc) for your work using the principles discussed earlier in the class. You must also include exception handling in your code according to the principles previously discussed and explored in more detail this week.

#Be sure to put comments in your code that clearly mark how you are performing your program logic. In the submission comments of this assignment, please place the repository URL of your file submission.

import os.path
import shutil
import json
