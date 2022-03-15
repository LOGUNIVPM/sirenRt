# SirenRT 0.1
# Siren Real-Time classification demo app
# Author: L. Gabrielli <l.gabrielli@staff.univpm.it>
# Date: Jan 2022 -

# PROJECT-WIDE DEFINES
TOUCHSCREENSIZE = (1280,800)

# LIBRARIES IMPORT
import sys
import os
import threading
import kivy
from kivy.app import App
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.utils import get_color_from_hex

from kivy.config import Config
#Config.set('graphics', 'fullscreen', 'fake')
Config.set('graphics', 'position', 'custom')
Config.set('graphics', 'top', '0')
Config.set('graphics', 'left', '0')
#Config.set('graphics', 'width', TOUCHSCREENSIZE[0])
#Config.set('graphics', 'height', TOUCHSCREENSIZE[1])
# pynstaller is unable to find imports in the kv file: add them here
from kivy.core.window import Window
Window.size = (TOUCHSCREENSIZE[0], TOUCHSCREENSIZE[1])
Window.borderless = True

# CUSTOM CLASSES IMPORT
from audioThread import audioThread

# APP
class SirenRtApp(App):
    isRunning = False

    def build(self):
        # the root is created in sirenrt.kv
        root = self.root
        print(root)

        # GUTS
        self.AT = audioThread()
        self.AT.fillAudioDevList()

        ### OGGETTI STATICI
        logo1 = Image(source='img/logo-w-alpha.png', size_hint=(None,None))
        orig_size = logo1.texture_size
        desiredW1 = 160
        scalefactor = desiredW1 / orig_size[0]
        logo1.size = (desiredW1,orig_size[1]*scalefactor)
        logo1.pos = (50, TOUCHSCREENSIZE[1]-50-logo1.size[1])
        root.add_widget(logo1)

        logo1 = Image(source='img/ilabs-logo.png', size_hint=(None,None))
        orig_size = logo1.texture_size
        desiredW2 = 220
        scalefactor = desiredW2 / orig_size[0]
        logo1.size = (desiredW2,orig_size[1]*scalefactor)
        logo1.pos = (50+desiredW1, TOUCHSCREENSIZE[1]-50-logo1.size[1])
        root.add_widget(logo1)

        lbltit2 = Label(text=u'SIREN RECOGNITION APP', font_name='fonts/ZillaSlab-Medium.ttf', font_size='70', halign="left")
        lbltit2.pos = (400, 600)
        lbltit2.bind(size=lbltit2.setter('text_size'))
        root.add_widget(lbltit2)

        lblbox1 = Label(text=u'AUDIO SETUP', font_name='fonts/ZillaSlab-Medium.ttf', font_size='20', halign="left")
        lblbox1.pos = (50, 500-20)
        lblbox1.bind(size=lblbox1.setter('text_size'))
        root.add_widget(lblbox1)

        lblbox1 = Label(text=u'DEMO', font_name='fonts/ZillaSlab-Medium.ttf', font_size='20', halign="left")
        lblbox1.pos = (600, 500-20)
        lblbox1.bind(size=lblbox1.setter('text_size'))
        root.add_widget(lblbox1)

        ### DYNAMIC OBJECTS

        # Audio device list dropdown
        soundcardDropdown = DropDown() # a dropdown of buttons
        for dev in audioThread.audioDevList:
            # When adding widgets, we need to specify the height manually
            # (disabling the size_hint_y) so the dropdown can calculate
            # the area it needs.
            btn = Button(text=dev, size_hint_y=None, height=30)
            # for each button, attach a callback that will call the select() method
            # on the dropdown. We'll pass the text of the button as the data of the
            # selection.
            btn.bind(on_press=self.cardSelectCallback)
            btn.bind(on_release=lambda btn: soundcardDropdown.select(btn.text))
            # then add the button inside the dropdown
            soundcardDropdown.add_widget(btn)

        # create a big main button
        mainbutton = Button(text='Select Sound Card First', size_hint=(0.3, 0.1))
        mainbutton.pos = (50, 350)

        # show the dropdown menu when the main button is released
        # note: all the bind() calls pass the instance of the caller (here, the
        # mainbutton instance) as the first argument of the callback (here,
        # dropdown.open.).
        mainbutton.bind(on_release=soundcardDropdown.open)

        # one last thing, listen for the selection in the dropdown list and
        # assign the data to the button text.
        soundcardDropdown.bind(on_select=lambda instance, x: setattr(mainbutton, 'text', x))
        root.add_widget(mainbutton)

        recordButton = Button(text='REC', size_hint=(0.1, 0.1))
        recordButton.pos = (50, 250)
        recordButton.bind(on_release=self.AT.recordOnce)
        root.add_widget(recordButton)

        playbackButton = Button(text='PLAY', size_hint=(0.1, 0.1))
        playbackButton.pos = (200, 250)
        playbackButton.bind(on_release=self.AT.playbackOnce)
        root.add_widget(playbackButton)

        exitButton = Button(text='EXIT', size_hint=(0.1, 0.1))
        exitButton.pos = (50, 50)
        exitButton.bind(on_release=self.exit)
        root.add_widget(exitButton)

        startButton = Button(text='START DEMO', size_hint=(0.15, 0.1))
        startButton.pos = (600, 350)
        startButton.bind(on_release=self.startDemo)
        root.add_widget(startButton)

        stopButton = Button(text='STOP DEMO', size_hint=(0.15, 0.1))
        stopButton.pos = (800, 350)
        stopButton.bind(on_release=self.stopDemo)
        root.add_widget(stopButton)

        # SIREN LABEL (TODO: make it an image)
        self.lblSiren = Label(text=u'--', font_name='fonts/ZillaSlab-Medium.ttf', font_size='70', halign="left")
        self.lblSiren.pos = (820, 180)
        self.lblSiren.bind(size=self.lblSiren.setter('text_size'))
        root.add_widget(self.lblSiren)

        # ALERT IMAGE
        self.imgSiren = Image(source='img/gray-light-192.png')
        self.imgSiren.pos = (600, 130)
        self.imgSiren.size_hint = (None, None)
        self.imgSiren.size = self.imgSiren.texture_size
        root.add_widget(self.imgSiren)


    def exit(self, instance):
        if self.isRunning:
            self.stopDemo()
        exit()


    def startDemo(self, instance):
        # REGULARLY CHECK ALARM
        self.clk = Clock.schedule_interval(lambda dt: self.checkAlarm(), 0.1)
        self.th = threading.Thread(target=self.AT.launch)
        self.th.start()
        self.isRunning = True

    def stopDemo(self, instance):
        self.AT.terminate()
        self.th.join()
        self.clk.cancel()
        self.lblSiren.text = '--'
        self.imgSiren.source = 'img/gray-light-192.png'
        self.isRunning = False

    def checkAlarm(self):
        if self.AT.statusChanged:
            if self.AT.alarm == True:
                self.lblSiren.text = 'SIREN!!'
                self.imgSiren.source = 'img/red-light-192.png'
            else:
                self.lblSiren.text = 'GO'
                self.imgSiren.source = 'img/green-light-192.png'

    def cardSelectCallback(self, instance):
        name = instance.text
        print(name)
        self.AT.initAudio(name)


def resourcePath():
    '''Returns path containing content - either locally or in pyinstaller tmp file'''
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS)

    return os.path.join(os.path.abspath("."))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # TODO: get device list and graphically select
    # interface: menu tendina con selezione audio dev e pulsante start/stop

    kivy.resources.resource_add_path(resourcePath())
    SirenRtApp().run()
