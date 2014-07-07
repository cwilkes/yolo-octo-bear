import sys

def lgi(msg, *args):
    print >>sys.stderr, msg % args
    sys.stderr.flush()
