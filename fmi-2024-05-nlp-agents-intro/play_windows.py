import ctypes, subprocess
from platform import system


def windowsCommand(command):
    ctypes.windll.winmm.mciSendStringW(command, ctypes.create_unicode_buffer(600), 559, 0);


def play(fileName):
    operatingSystem = system();

    if (operatingSystem == "Windows"):
        windowsCommand("open " + fileName);
        windowsCommand("play " + fileName + " wait");
        windowsCommand("close " + fileName);
    else:
        subprocess.Popen(
            operatingSystem == "Darwin" and "exec afplay \"" + fileName + "\"" or operatingSystem == "Linux" and "exec aplay --quiet " + fileName,
            universal_newlines=True, shell=True, stdout=-1, stderr=-1).communicate();
