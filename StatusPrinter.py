import datetime

class StatusPrinter:
    def __init__(self):
        self.level = 0

    def Indent(self):
        self.level = self.level + 1

    def Unindent(self):
        if(self.level > 0):
            self.level = self.level - 1

    def Print(self, message, prependTimestamp = True):
        prefix = ""

        if(prependTimestamp):
            prefix = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        else:
            prefix = " " * 21

        print("{} {}{}".format(prefix, " " * (self.level * 2), message))