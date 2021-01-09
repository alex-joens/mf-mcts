from os import fdopen
import sys
from queue import Empty
import time

# Generic message class
class Message:
    def __init__(self, cmd, args):
        self.cmd = cmd
        self.args = args

    # For MessageReader
    @classmethod
    def fromText(cls, line):
        if line is None:
            return None
        args = line.split(' ')
        cmd = args[0]
        args = args[1:]
        return cls(cmd, args)

    # Standard response for mode change and init
    @classmethod
    def ack(cls):
        return cls("K", None)

    # Used during preparation and training
    @classmethod
    def requestSample(cls, fidelity, actionSequence):
        if actionSequence is None:
            return cls("S", [fidelity])
        return cls("S", [fidelity] + actionSequence)

    # Used during testing
    @classmethod
    def action(cls, action, options):
        return cls("A", [action] + options)

    # For MessageWriter
    def toText(self):
        return self.cmd + (' ' + ' '.join(map(str, self.args)) if self.args is not None else "")


# Manages all messaging
class Handler:
    def __init__(self, msgQ, readTimeout):
        self.timeout = readTimeout
        self.inQ = msgQ

    def ack(self):
        self.send(Message.ack())

    def act(self, action, options):
        self.send(Message.action(action, options))

    def get(self):
        try:
            line = self.inQ.get(True, self.timeout)
            return Message.fromText(line)
        except Empty:
            self.error("timeout")
            raise Exception        

    def send(self, message):
        print(message.toText(), flush=True, file=sys.stdout)

    def error(self, errorText):
        print("E " + errorText, flush=True, file=sys.stderr)


# Must be directly instantiated by __main__  by 'multiprocessing', because of reasons
def LineReader(msgQ, fileno):
    sys.stdin = fdopen(fileno)
    while True:
        line = input().rstrip('\r\n')
        if len(line) > 0:
            msgQ.put(line)