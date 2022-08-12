import mowl
import jpype

if not jpype.isJVMStarted():
    mowl.init_jvm("5g")

